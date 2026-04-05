import math
import os
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from types import MethodType

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.operations import ConvertOutputsToFp32
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.config import Stage1Config
from src.grpo import (
    PackedRolloutTensors,
    build_loss_normalizer,
    compute_advantages,
    compute_completion_logprobs,
    compute_policy_objective,
    pack_rollout_sequences,
    reduce_policy_loss,
)
from src.modeling import count_trainable_parameters, save_adapter_checkpoint
from src.retro_rollout import RolloutBatch, run_baseline_rollouts, run_retro_rollouts


@dataclass(slots=True)
class PreparedRolloutBatch:
    rollout: RolloutBatch
    packed: PackedRolloutTensors
    old_logprobs: torch.Tensor
    completion_mask: torch.Tensor
    ref_logprobs: torch.Tensor | None
    advantages: torch.Tensor | None = None


class Stage1Trainer:
    rollout_label = "baseline"

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        config: Stage1Config,
        run_name: str,
        output_dir: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.config = config
        self.run_name = run_name
        self.output_dir = output_dir
        self.global_step = 0

        mixed_precision = (
            "bf16" if config.dtype == "bfloat16" else "fp16" if config.dtype == "float16" else "no"
        )
        self.accelerator = Accelerator(log_with="wandb", mixed_precision=mixed_precision)
        set_seed(config.seed)

        self.optimizer = AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            collate_fn=list,
        )

        total_optimizer_steps = self._estimate_total_optimizer_steps()
        self.scheduler = get_scheduler(
            config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )
        self._disable_output_widening()

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=config.to_dict(),
                init_kwargs={"wandb": {"name": run_name}},
            )
            trainable, total = count_trainable_parameters(
                self.accelerator.unwrap_model(self.model)
            )
            self.accelerator.log(
                {
                    "model/trainable_params": float(trainable),
                    "model/total_params": float(total),
                    "model/trainable_fraction": trainable / total if total else 0.0,
                },
                step=0,
            )

    def _estimate_total_optimizer_steps(self) -> int:
        micro_steps_per_epoch = math.ceil(
            len(self.train_dataset) / self.config.per_device_train_batch_size
        )
        optimizer_steps_per_epoch = math.ceil(
            micro_steps_per_epoch / self.config.gradient_accumulation_steps
        )
        return max(optimizer_steps_per_epoch * self.config.num_train_epochs, 1)

    def _disable_output_widening(self) -> None:
        base_model = self.accelerator.unwrap_model(self.model)
        wrapped = getattr(base_model.forward, "__wrapped__", None)
        if wrapped is None:
            return
        if not isinstance(wrapped, ConvertOutputsToFp32):
            return

        autocast_forward = getattr(wrapped, "__wrapped__", None)
        if autocast_forward is None:
            raise RuntimeError("Unexpected Accelerate forward wrapper chain.")
        base_model.forward = MethodType(autocast_forward, base_model)

    def _synchronize_device(self) -> None:
        if self.accelerator.device.type == "cuda":
            torch.cuda.synchronize(self.accelerator.device)

    def _generate_rollout_batch(
        self, unwrapped_model, prompt_batch: list[list[dict]], answers: list[str]
    ) -> RolloutBatch:
        raise NotImplementedError

    def _collect_window(self, iterator) -> list[list[dict]]:
        microbatches = []
        for _ in range(self.config.gradient_accumulation_steps):
            try:
                microbatches.append(next(iterator))
            except StopIteration:
                break
        return microbatches

    def _prepare_window(
        self, microbatches: list[list[dict]]
    ) -> tuple[list[PreparedRolloutBatch], dict[str, float]]:
        prepared_batches: list[PreparedRolloutBatch] = []
        prep_metrics = {
            "time/old_logprobs_s": 0.0,
            "time/ref_logprobs_s": 0.0,
            "tokens/old_logprob_total": 0.0,
        }
        group_offset = 0
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.eval()

        for batch_rows in microbatches:
            prompt_batch = [row["prompt"] for row in batch_rows]
            answers = [row["final_answer"] for row in batch_rows]
            rollout = self._generate_rollout_batch(unwrapped_model, prompt_batch, answers)
            rollout.group_ids = [group_id + group_offset for group_id in rollout.group_ids]
            group_offset += len(prompt_batch)

            packed = pack_rollout_sequences(
                sequences=rollout.sequences,
                prompt_lengths=rollout.prompt_lengths,
                pad_token_id=self.tokenizer.pad_token_id,
                device=self.accelerator.device,
            )
            self._synchronize_device()
            old_logprobs_start = perf_counter()
            with torch.no_grad():
                old_logprobs, completion_mask = compute_completion_logprobs(
                    unwrapped_model, packed
                )
            self._synchronize_device()
            prep_metrics["time/old_logprobs_s"] += perf_counter() - old_logprobs_start

            if self.config.mask_truncated_completions:
                truncated_mask = torch.tensor(
                    rollout.truncated,
                    dtype=torch.bool,
                    device=self.accelerator.device,
                )
                completion_mask = completion_mask & ~truncated_mask.unsqueeze(1)
                old_logprobs = torch.where(
                    completion_mask, old_logprobs, torch.zeros_like(old_logprobs)
                )

            ref_logprobs = None
            if self.config.beta > 0.0:
                if not hasattr(unwrapped_model, "disable_adapter"):
                    raise ValueError(
                        "beta > 0 requires a PEFT model so the frozen base policy can be used as reference."
                    )
                with unwrapped_model.disable_adapter():
                    self._synchronize_device()
                    ref_logprobs_start = perf_counter()
                    with torch.no_grad():
                        ref_logprobs, _ = compute_completion_logprobs(unwrapped_model, packed)
                    self._synchronize_device()
                    prep_metrics["time/ref_logprobs_s"] += perf_counter() - ref_logprobs_start
                ref_logprobs = torch.where(
                    completion_mask, ref_logprobs, torch.zeros_like(ref_logprobs)
                )

            prep_metrics["tokens/old_logprob_total"] += float(completion_mask.sum().item())

            prepared_batches.append(
                PreparedRolloutBatch(
                    rollout=rollout,
                    packed=packed,
                    old_logprobs=old_logprobs.detach(),
                    completion_mask=completion_mask.detach(),
                    ref_logprobs=None if ref_logprobs is None else ref_logprobs.detach(),
                )
            )

        unwrapped_model.train()
        return prepared_batches, prep_metrics

    def _assign_advantages(
        self, prepared_batches: list[PreparedRolloutBatch]
    ) -> tuple[object, dict[str, float]]:
        rewards = torch.cat(
            [
                torch.tensor(batch.rollout.rewards, dtype=torch.float32, device=self.accelerator.device)
                for batch in prepared_batches
            ]
        )
        group_ids = torch.cat(
            [
                torch.tensor(batch.rollout.group_ids, dtype=torch.long, device=self.accelerator.device)
                for batch in prepared_batches
            ]
        )

        advantages, stats = compute_advantages(
            rewards=rewards,
            group_ids=group_ids,
            scale_rewards=self.config.scale_rewards,
        )
        normalizer = build_loss_normalizer(
            completion_masks=[batch.completion_mask for batch in prepared_batches],
            loss_type=self.config.loss_type,
            max_completion_length=self.config.max_completion_length,
        )

        cursor = 0
        for batch in prepared_batches:
            batch_size = len(batch.rollout.rewards)
            batch.advantages = advantages[cursor : cursor + batch_size]
            cursor += batch_size

        return normalizer, stats

    def _sync_context(self, batch_index: int, total_batches: int):
        if (
            self.accelerator.num_processes > 1
            and batch_index < total_batches - 1
            and hasattr(self.model, "no_sync")
        ):
            return self.model.no_sync()
        return nullcontext()

    def _slice_packed_batch(
        self, packed: PackedRolloutTensors, start: int, stop: int
    ) -> PackedRolloutTensors:
        input_ids = packed.input_ids[start:stop]
        attention_mask = packed.attention_mask[start:stop]
        prompt_lengths = packed.prompt_lengths[start:stop]

        active_columns = attention_mask.any(dim=0)
        if active_columns.any():
            first_active_column = int(active_columns.nonzero(as_tuple=False)[0].item())
            input_ids = input_ids[:, first_active_column:]
            attention_mask = attention_mask[:, first_active_column:]

        return PackedRolloutTensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )

    def _iter_update_subbatches(self, batch: PreparedRolloutBatch):
        prompt_chunk_size = max(self.config.update_prompt_microbatch_size, 1)
        sequence_chunk_size = prompt_chunk_size * self.config.num_generations
        total_sequences = batch.packed.input_ids.shape[0]

        for start in range(0, total_sequences, sequence_chunk_size):
            stop = min(start + sequence_chunk_size, total_sequences)
            packed = self._slice_packed_batch(batch.packed, start, stop)
            yield start, stop, packed

    def _summarize_rollout_metrics(
        self, prepared_batches: list[PreparedRolloutBatch]
    ) -> dict[str, float]:
        raw: dict[str, float] = {}
        for batch in prepared_batches:
            for key, value in batch.rollout.metrics.items():
                raw[key] = raw.get(key, 0.0) + float(value)

        metrics: dict[str, float] = {}
        rollout_labels = [
            label
            for label in ("baseline", "retro_scout", "retro_conditioned")
            if raw.get(f"{label}_count", 0.0) > 0.0
        ]
        generated_total = sum(raw.get(f"{label}_completion_tokens", 0.0) for label in rollout_labels)
        generated_count = sum(raw.get(f"{label}_count", 0.0) for label in rollout_labels)
        generated_max = max(
            (raw.get(f"{label}_completion_tokens_max", 0.0) for label in rollout_labels),
            default=0.0,
        )
        prompt_total = sum(raw.get(f"{label}_prompt_tokens", 0.0) for label in rollout_labels)
        prompt_max = max(
            (raw.get(f"{label}_prompt_tokens_max", 0.0) for label in rollout_labels),
            default=0.0,
        )
        sequence_total = sum(raw.get(f"{label}_sequence_tokens", 0.0) for label in rollout_labels)
        sequence_max = max(
            (raw.get(f"{label}_sequence_tokens_max", 0.0) for label in rollout_labels),
            default=0.0,
        )
        rollout_tokenize_s = sum(raw.get(f"{label}_tokenize_s", 0.0) for label in rollout_labels)
        rollout_generate_s = sum(raw.get(f"{label}_generate_s", 0.0) for label in rollout_labels)
        rollout_postprocess_s = sum(
            raw.get(f"{label}_postprocess_s", 0.0) for label in rollout_labels
        )
        reward_score_s = sum(raw.get(f"{label}_reward_s", 0.0) for label in rollout_labels)

        metrics["time/rollout_tokenize_s"] = rollout_tokenize_s
        metrics["time/rollout_generate_s"] = rollout_generate_s
        metrics["time/rollout_postprocess_s"] = rollout_postprocess_s
        metrics["time/reward_score_s"] = reward_score_s
        metrics["time/summary_generate_s"] = raw.get("retro_summary_generate_s", 0.0)
        metrics["tokens/generated_total"] = generated_total
        metrics["tokens/generated_mean"] = (
            generated_total / generated_count if generated_count > 0.0 else 0.0
        )
        metrics["tokens/generated_max"] = generated_max
        metrics["tokens/prompt_total"] = prompt_total
        metrics["tokens/prompt_mean"] = (
            prompt_total / generated_count if generated_count > 0.0 else 0.0
        )
        metrics["tokens/prompt_max"] = prompt_max
        metrics["tokens/sequence_total"] = sequence_total
        metrics["tokens/sequence_mean"] = (
            sequence_total / generated_count if generated_count > 0.0 else 0.0
        )
        metrics["tokens/sequence_max"] = sequence_max
        metrics["tokens/summary_total"] = raw.get("retro_summary_tokens", 0.0)
        metrics["tokens/summary_mean"] = (
            raw.get("retro_summary_tokens", 0.0) / raw["retro_summary_count"]
            if raw.get("retro_summary_count", 0.0) > 0.0
            else 0.0
        )
        metrics["tokens/summary_max"] = raw.get("retro_summary_tokens_max", 0.0)
        metrics["throughput/generated_tokens_per_s"] = (
            generated_total / rollout_generate_s if rollout_generate_s > 0.0 else 0.0
        )
        metrics["throughput/summary_tokens_per_s"] = (
            raw.get("retro_summary_tokens", 0.0) / raw["retro_summary_generate_s"]
            if raw.get("retro_summary_generate_s", 0.0) > 0.0
            else 0.0
        )

        if raw.get("baseline_count", 0.0) > 0.0:
            metrics["baseline/solve_rate"] = raw["baseline_correct"] / raw["baseline_count"]
            metrics["baseline/truncated_rate"] = raw["baseline_truncated"] / raw["baseline_count"]
            metrics["baseline/mean_completion_length"] = (
                raw["baseline_completion_tokens"] / raw["baseline_count"]
            )
            metrics["baseline/generate_s"] = raw.get("baseline_generate_s", 0.0)
            metrics["baseline/prompt_tokens_mean"] = (
                raw["baseline_prompt_tokens"] / raw["baseline_count"]
            )
            metrics["baseline/completion_tokens_max"] = raw.get(
                "baseline_completion_tokens_max", 0.0
            )

        if raw.get("retro_scout_count", 0.0) > 0.0:
            metrics["retro/scout_solve_rate"] = raw["retro_scout_correct"] / raw["retro_scout_count"]
            metrics["retro/scout_truncated_rate"] = (
                raw["retro_scout_truncated"] / raw["retro_scout_count"]
            )
            metrics["retro/scout_mean_completion_length"] = (
                raw["retro_scout_completion_tokens"] / raw["retro_scout_count"]
            )
            metrics["retro/scout_generate_s"] = raw.get("retro_scout_generate_s", 0.0)
            metrics["retro/scout_completion_tokens_max"] = raw.get(
                "retro_scout_completion_tokens_max", 0.0
            )

        if raw.get("retro_conditioned_count", 0.0) > 0.0:
            metrics["retro/conditioned_solve_rate"] = (
                raw["retro_conditioned_correct"] / raw["retro_conditioned_count"]
            )
            metrics["retro/conditioned_truncated_rate"] = (
                raw["retro_conditioned_truncated"] / raw["retro_conditioned_count"]
            )
            metrics["retro/conditioned_mean_completion_length"] = (
                raw["retro_conditioned_completion_tokens"]
                / raw["retro_conditioned_count"]
            )
            metrics["retro/conditioned_generate_s"] = raw.get(
                "retro_conditioned_generate_s", 0.0
            )
            metrics["retro/conditioned_completion_tokens_max"] = raw.get(
                "retro_conditioned_completion_tokens_max", 0.0
            )

        if raw.get("retro_prompt_count", 0.0) > 0.0:
            metrics["retro/scheduled_conditioning_rate"] = 1.0
            metrics["retro/failure_context_rate"] = (
                raw["retro_summary_prompt_count"] / raw["retro_prompt_count"]
            )
            metrics["retro/summary_generate_s"] = raw.get("retro_summary_generate_s", 0.0)
            metrics["retro/summary_mean_tokens"] = (
                raw.get("retro_summary_tokens", 0.0) / raw["retro_summary_count"]
                if raw.get("retro_summary_count", 0.0) > 0.0
                else 0.0
            )
            metrics["retro/failed_rollout_count"] = raw.get("retro_failed_rollout_count", 0.0)

        return metrics

    def _apply_window_update(
        self,
        prepared_batches: list[PreparedRolloutBatch],
        normalizer,
    ) -> dict[str, float]:
        self._synchronize_device()
        update_start = perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_ratio = 0.0
        total_clip_fraction = 0.0
        total_kl = 0.0

        for batch_index, batch in enumerate(prepared_batches):
            with self._sync_context(batch_index, len(prepared_batches)):
                batch_loss = 0.0
                batch_ratio_numerator = 0.0
                batch_clip_numerator = 0.0
                batch_kl_numerator = 0.0
                batch_metric_denominator = 0

                for start, stop, packed in self._iter_update_subbatches(batch):
                    current_logprobs, _ = compute_completion_logprobs(self.model, packed)
                    seq_len = current_logprobs.shape[1]
                    completion_mask = batch.completion_mask[start:stop, -seq_len:]
                    old_logprobs = batch.old_logprobs[start:stop, -seq_len:]
                    ref_logprobs = (
                        None
                        if batch.ref_logprobs is None
                        else batch.ref_logprobs[start:stop, -seq_len:]
                    )
                    advantages = batch.advantages[start:stop]

                    per_token_objective, objective_metrics = compute_policy_objective(
                        current_logprobs=current_logprobs,
                        old_logprobs=old_logprobs,
                        advantages=advantages,
                        completion_mask=completion_mask,
                        epsilon=self.config.epsilon,
                        epsilon_high=self.config.epsilon_high,
                        ref_logprobs=ref_logprobs,
                        beta=self.config.beta,
                    )
                    loss = reduce_policy_loss(
                        per_token_objective=per_token_objective,
                        completion_mask=completion_mask,
                        normalizer=normalizer,
                    )
                    self.accelerator.backward(loss)

                    batch_loss += loss.detach().item()
                    active_tokens = int(completion_mask.sum().item())
                    if active_tokens > 0:
                        batch_ratio_numerator += objective_metrics["ratio_mean"] * active_tokens
                        batch_clip_numerator += (
                            objective_metrics["clip_fraction"] * active_tokens
                        )
                        batch_kl_numerator += objective_metrics["kl_mean"] * active_tokens
                        batch_metric_denominator += active_tokens

            total_loss += batch_loss
            if batch_metric_denominator > 0:
                total_ratio += batch_ratio_numerator / batch_metric_denominator
                total_clip_fraction += batch_clip_numerator / batch_metric_denominator
                total_kl += batch_kl_numerator / batch_metric_denominator
            else:
                total_ratio += 1.0
                total_clip_fraction += 0.0
                total_kl += 0.0

        if self.config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._synchronize_device()
        update_s = perf_counter() - update_start

        denom = max(len(prepared_batches), 1)
        return {
            "loss/policy": total_loss / denom,
            "loss/ratio_mean": total_ratio / denom,
            "loss/clip_fraction": total_clip_fraction / denom,
            "loss/kl_mean": total_kl / denom,
            "lr": self.scheduler.get_last_lr()[0],
            "time/update_s": update_s,
            "throughput/update_active_tokens_per_s": (
                normalizer.total_active_tokens / update_s if update_s > 0.0 else 0.0
            ),
        }

    def _save_checkpoint(self, path: str, metadata: dict[str, float] | None = None) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_adapter_checkpoint(
                model=unwrapped_model,
                tokenizer=self.tokenizer,
                output_dir=path,
                config=self.config,
                metadata=metadata,
            )

    def _prune_old_checkpoints(self) -> None:
        if self.config.save_total_limit <= 0 or not os.path.isdir(self.output_dir):
            return

        checkpoint_dirs = sorted(
            [
                os.path.join(self.output_dir, name)
                for name in os.listdir(self.output_dir)
                if name.startswith("checkpoint-")
            ]
        )
        while len(checkpoint_dirs) > self.config.save_total_limit:
            stale = checkpoint_dirs.pop(0)
            shutil.rmtree(stale, ignore_errors=True)

    def train(self, max_steps: int = -1) -> None:
        optimizer_steps_target = (
            max_steps if max_steps > 0 else self._estimate_total_optimizer_steps()
        )
        self.accelerator.print(
            f"Starting {self.rollout_label} training for up to {optimizer_steps_target} optimizer steps."
        )

        for epoch in range(self.config.num_train_epochs):
            iterator = iter(self.train_dataloader)
            while True:
                microbatches = self._collect_window(iterator)
                if not microbatches:
                    break

                if self.accelerator.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.accelerator.device)
                self._synchronize_device()
                optimizer_step_start = perf_counter()

                prepare_window_start = perf_counter()
                prepared_batches, prepare_metrics = self._prepare_window(microbatches)
                prepare_window_s = perf_counter() - prepare_window_start

                advantage_start = perf_counter()
                normalizer, reward_stats = self._assign_advantages(prepared_batches)
                advantage_s = perf_counter() - advantage_start
                update_metrics = self._apply_window_update(prepared_batches, normalizer)
                rollout_metrics = self._summarize_rollout_metrics(prepared_batches)
                self._synchronize_device()
                optimizer_step_s = perf_counter() - optimizer_step_start
                unaccounted_step_s = max(
                    optimizer_step_s
                    - prepare_window_s
                    - advantage_s
                    - update_metrics["time/update_s"],
                    0.0,
                )

                self.global_step += 1
                metrics = {
                    "train/epoch": epoch + 1,
                    "train/optimizer_step": self.global_step,
                    "train/reward_mean": reward_stats["reward_mean"],
                    "train/reward_std": reward_stats["reward_std"],
                    "train/zero_std_fraction": reward_stats["zero_std_fraction"],
                    "train/active_tokens": float(normalizer.total_active_tokens),
                    "train/num_sequences": float(normalizer.num_sequences),
                    "time/prepare_window_s": prepare_window_s,
                    "time/advantage_s": advantage_s,
                    "time/optimizer_step_s": optimizer_step_s,
                    "time/unaccounted_step_s": unaccounted_step_s,
                    "throughput/optimizer_steps_per_hour": (
                        3600.0 / optimizer_step_s if optimizer_step_s > 0.0 else 0.0
                    ),
                    "throughput/old_logprob_tokens_per_s": (
                        prepare_metrics["tokens/old_logprob_total"]
                        / prepare_metrics["time/old_logprobs_s"]
                        if prepare_metrics["time/old_logprobs_s"] > 0.0
                        else 0.0
                    ),
                    **update_metrics,
                    **prepare_metrics,
                    **rollout_metrics,
                }
                if self.accelerator.device.type == "cuda":
                    metrics["gpu/max_memory_allocated_gb"] = (
                        torch.cuda.max_memory_allocated(self.accelerator.device) / (1024**3)
                    )
                    metrics["gpu/max_memory_reserved_gb"] = (
                        torch.cuda.max_memory_reserved(self.accelerator.device) / (1024**3)
                    )

                if (
                    self.accelerator.is_main_process
                    and self.global_step % self.config.logging_steps == 0
                ):
                    self.accelerator.log(metrics, step=self.global_step)

                if (
                    self.global_step % self.config.save_steps == 0
                    and self.accelerator.is_main_process
                ):
                    checkpoint_dir = os.path.join(
                        self.output_dir, f"checkpoint-{self.global_step}"
                    )
                    self._save_checkpoint(checkpoint_dir, metadata=metrics)
                    self._prune_old_checkpoints()

                if max_steps > 0 and self.global_step >= max_steps:
                    self.accelerator.wait_for_everyone()
                    self.accelerator.end_training()
                    return

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


class BaselineTrainer(Stage1Trainer):
    rollout_label = "baseline"

    def _generate_rollout_batch(
        self, unwrapped_model, prompt_batch: list[list[dict]], answers: list[str]
    ) -> RolloutBatch:
        return run_baseline_rollouts(
            model=unwrapped_model,
            tokenizer=self.tokenizer,
            prompt_messages_batch=prompt_batch,
            answers=answers,
            config=self.config,
        )


class RetroTrainer(Stage1Trainer):
    rollout_label = "retro"

    def _generate_rollout_batch(
        self, unwrapped_model, prompt_batch: list[list[dict]], answers: list[str]
    ) -> RolloutBatch:
        return run_retro_rollouts(
            model=unwrapped_model,
            tokenizer=self.tokenizer,
            prompt_messages_batch=prompt_batch,
            answers=answers,
            config=self.config,
        )
