"""Custom rollout engines for baseline GRPO and long-memory RETRO-GRPO."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from time import perf_counter

import torch

from src.config import Stage1Config
from src.memory import (
    MemoryRecord,
    build_memory_record,
    compute_failure_mode_overlap,
    decay_memory_record,
    extract_failure_modes,
    should_retire_record,
    update_solve_rate_ema,
)
from src.reward import score_completion_texts
from src.summariser import (
    FailureSummaryRequest,
    MemoryRefreshRequest,
    build_conditioned_prompt,
    generate_failure_summaries,
    generate_refreshed_memories,
    wrap_in_framing,
)
from src.token_utils import truncate_from_left


@dataclass(slots=True)
class RolloutBatch:
    sequences: list[list[int]]
    prompt_lengths: list[int]
    completion_ids: list[list[int]]
    completion_texts: list[str]
    rewards: list[float]
    group_ids: list[int]
    truncated: list[bool]
    metrics: dict[str, float] = field(default_factory=dict)


def _synchronize_model_device(model) -> None:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _trim_completion_ids(
    token_ids: list[int],
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> tuple[list[int], bool]:
    if eos_token_id is not None and eos_token_id in token_ids:
        eos_index = token_ids.index(eos_token_id)
        return token_ids[: eos_index + 1], False

    if pad_token_id is not None and pad_token_id != eos_token_id:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()

    return token_ids, True


def _clip_text_to_token_budget(text: str, tokenizer, max_tokens: int) -> str:
    if not text.strip():
        return ""
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) <= max_tokens:
        return text.strip()
    clipped_ids = token_ids[:max_tokens]
    return tokenizer.decode(clipped_ids, skip_special_tokens=True).strip()


def _group_completion_texts_by_prompt(
    completion_texts: list[str],
    rewards: list[float],
    group_ids: list[int],
    num_prompts: int,
) -> list[list[str]]:
    grouped = [[] for _ in range(num_prompts)]
    for completion_text, reward, group_id in zip(completion_texts, rewards, group_ids):
        if reward == 0.0:
            grouped[group_id].append(completion_text)
    return grouped


def _group_rewards_by_prompt(
    rewards: list[float],
    group_ids: list[int],
    num_prompts: int,
) -> list[list[float]]:
    grouped = [[] for _ in range(num_prompts)]
    for reward, group_id in zip(rewards, group_ids):
        grouped[group_id].append(reward)
    return grouped


@torch.no_grad()
def _generate_policy_rollouts(
    model,
    tokenizer,
    prompt_messages_batch: list[list[dict]],
    answers: list[str],
    config: Stage1Config,
    label: str,
) -> RolloutBatch:
    tokenize_start = perf_counter()
    max_prompt_tokens = max(config.max_seq_length - config.max_completion_length, 1)
    prompt_token_lists = [
        truncate_from_left(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=config.rollout_enable_thinking,
            ),
            max_prompt_tokens,
        )
        for messages in prompt_messages_batch
    ]
    model_inputs = tokenizer.pad(
        {"input_ids": prompt_token_lists},
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    expanded_input_ids = model_inputs["input_ids"].repeat_interleave(
        config.num_generations, dim=0
    )
    expanded_attention_mask = model_inputs["attention_mask"].repeat_interleave(
        config.num_generations, dim=0
    )
    tokenize_s = perf_counter() - tokenize_start

    _synchronize_model_device(model)
    generate_start = perf_counter()
    outputs = model.generate(
        input_ids=expanded_input_ids,
        attention_mask=expanded_attention_mask,
        max_new_tokens=config.max_completion_length,
        do_sample=True,
        temperature=config.rollout_temperature,
        top_p=config.rollout_top_p,
        top_k=config.rollout_top_k,
        min_p=config.rollout_min_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    _synchronize_model_device(model)
    generate_s = perf_counter() - generate_start

    postprocess_start = perf_counter()
    padded_prompt_length = expanded_input_ids.shape[1]
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    completion_ids: list[list[int]] = []
    completion_texts: list[str] = []
    group_ids: list[int] = []
    truncated: list[bool] = []

    for seq_idx in range(outputs.shape[0]):
        prompt_idx = seq_idx // config.num_generations
        prompt_ids = prompt_token_lists[prompt_idx]
        raw_completion_ids = outputs[seq_idx, padded_prompt_length:].tolist()
        trimmed_completion_ids, is_truncated = _trim_completion_ids(
            raw_completion_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        sequences.append(prompt_ids + trimmed_completion_ids)
        prompt_lengths.append(len(prompt_ids))
        completion_ids.append(trimmed_completion_ids)
        completion_texts.append(
            tokenizer.decode(trimmed_completion_ids, skip_special_tokens=True).strip()
        )
        group_ids.append(prompt_idx)
        truncated.append(is_truncated)
    postprocess_s = perf_counter() - postprocess_start

    reward_start = perf_counter()
    repeated_answers = [answers[group_id] for group_id in group_ids]
    rewards = score_completion_texts(completion_texts, repeated_answers)
    reward_s = perf_counter() - reward_start

    completion_lengths = [len(ids) for ids in completion_ids]
    prompt_token_counts = prompt_lengths
    sequence_lengths = [
        prompt + completion for prompt, completion in zip(prompt_lengths, completion_lengths)
    ]

    metrics = {
        f"{label}_correct": float(sum(rewards)),
        f"{label}_count": float(len(rewards)),
        f"{label}_truncated": float(sum(1 for item in truncated if item)),
        f"{label}_tokenize_s": tokenize_s,
        f"{label}_generate_s": generate_s,
        f"{label}_postprocess_s": postprocess_s,
        f"{label}_reward_s": reward_s,
        f"{label}_prompt_tokens": float(sum(prompt_token_counts)),
        f"{label}_prompt_tokens_max": (
            float(max(prompt_token_counts)) if prompt_token_counts else 0.0
        ),
        f"{label}_completion_tokens": float(sum(completion_lengths)),
        f"{label}_completion_tokens_max": (
            float(max(completion_lengths)) if completion_lengths else 0.0
        ),
        f"{label}_sequence_tokens": float(sum(sequence_lengths)),
        f"{label}_sequence_tokens_max": (
            float(max(sequence_lengths)) if sequence_lengths else 0.0
        ),
    }

    return RolloutBatch(
        sequences=sequences,
        prompt_lengths=prompt_lengths,
        completion_ids=completion_ids,
        completion_texts=completion_texts,
        rewards=rewards,
        group_ids=group_ids,
        truncated=truncated,
        metrics=metrics,
    )


@torch.no_grad()
def run_baseline_rollouts(
    model,
    tokenizer,
    prompt_messages_batch: list[list[dict]],
    answers: list[str],
    config: Stage1Config,
) -> RolloutBatch:
    return _generate_policy_rollouts(
        model=model,
        tokenizer=tokenizer,
        prompt_messages_batch=prompt_messages_batch,
        answers=answers,
        config=config,
        label="baseline",
    )


@torch.no_grad()
def run_retro_rollouts(
    model,
    tokenizer,
    prompt_messages_batch: list[list[dict]],
    answers: list[str],
    prompt_ids: list[str],
    prompt_solve_emas: dict[str, float],
    memory_store: dict[str, MemoryRecord],
    config: Stage1Config,
    global_step: int,
) -> RolloutBatch:
    """Generate scout rollouts, refresh prompt-local memories, and run conditioned rollouts."""
    scout_batch = _generate_policy_rollouts(
        model=model,
        tokenizer=tokenizer,
        prompt_messages_batch=prompt_messages_batch,
        answers=answers,
        config=config,
        label="retro_scout",
    )

    num_prompts = len(prompt_messages_batch)
    failed_rollouts_by_prompt = _group_completion_texts_by_prompt(
        scout_batch.completion_texts,
        scout_batch.rewards,
        scout_batch.group_ids,
        num_prompts=num_prompts,
    )
    existing_records = [memory_store.get(prompt_id) for prompt_id in prompt_ids]
    solve_rate_history = [prompt_solve_emas.get(prompt_id, 0.0) for prompt_id in prompt_ids]
    hard_prompt_mask = [
        solve_rate < config.memory_hard_prompt_threshold for solve_rate in solve_rate_history
    ]

    summary_requests: list[FailureSummaryRequest] = []
    summary_owner_indices: list[int] = []
    current_summaries: list[str | None] = [None] * num_prompts
    existing_prompt_count = 0
    current_failure_prompt_count = 0
    stale_prompt_count = 0

    for prompt_idx, prompt_messages in enumerate(prompt_messages_batch):
        if existing_records[prompt_idx] is not None:
            existing_prompt_count += 1
        failed_rollouts = failed_rollouts_by_prompt[prompt_idx]
        if failed_rollouts:
            current_failure_prompt_count += 1
        else:
            if existing_records[prompt_idx] is not None:
                stale_prompt_count += 1
            continue
        if not hard_prompt_mask[prompt_idx]:
            continue
        summary_requests.append(
            FailureSummaryRequest(
                question=prompt_messages[-1]["content"],
                failed_rollouts=failed_rollouts,
            )
        )
        summary_owner_indices.append(prompt_idx)

    summary_generate_s = 0.0
    if summary_requests:
        _synchronize_model_device(model)
        summary_start = perf_counter()
        summary_outputs = generate_failure_summaries(
            model=model,
            tokenizer=tokenizer,
            requests=summary_requests,
            rollout_summary_max_new_tokens=config.rollout_summary_max_new_tokens,
            aggregate_summary_max_new_tokens=config.aggregate_summary_max_new_tokens,
            enable_thinking=config.summary_enable_thinking,
            do_sample=config.summary_do_sample,
            summariser_mode=config.summariser_mode,
            max_seq_length=config.max_seq_length,
        )
        _synchronize_model_device(model)
        summary_generate_s = perf_counter() - summary_start
        for owner_idx, summary in zip(summary_owner_indices, summary_outputs):
            current_summaries[owner_idx] = summary

    refreshed_summaries: list[str | None] = current_summaries[:]
    refresh_requests: list[MemoryRefreshRequest] = []
    refresh_owner_indices: list[int] = []
    merged_prompt_count = 0

    for prompt_idx, current_summary in enumerate(current_summaries):
        existing_record = existing_records[prompt_idx]
        if existing_record is None or current_summary is None:
            continue
        refresh_requests.append(
            MemoryRefreshRequest(
                question=prompt_messages_batch[prompt_idx][-1]["content"],
                prior_memory=existing_record.summary_text,
                fresh_summary=current_summary,
            )
        )
        refresh_owner_indices.append(prompt_idx)
        merged_prompt_count += 1

    refresh_generate_s = 0.0
    if refresh_requests:
        _synchronize_model_device(model)
        refresh_start = perf_counter()
        refresh_outputs = generate_refreshed_memories(
            model=model,
            tokenizer=tokenizer,
            requests=refresh_requests,
            max_new_tokens=config.memory_refresh_max_new_tokens,
            enable_thinking=config.summary_enable_thinking,
            do_sample=config.summary_do_sample,
            summariser_mode=config.summariser_mode,
            max_seq_length=config.max_seq_length,
        )
        _synchronize_model_device(model)
        refresh_generate_s = perf_counter() - refresh_start
        for owner_idx, refreshed_summary in zip(refresh_owner_indices, refresh_outputs):
            refreshed_summaries[owner_idx] = (
                refreshed_summary or current_summaries[owner_idx]
            )

    stored_memory_texts: list[str | None] = [None] * num_prompts
    conditioned_messages_batch: list[list[dict]] = []
    injected_memory_texts: list[str] = []
    injected_prompt_count = 0
    dropped_prompt_count = 0

    for prompt_idx, prompt_messages in enumerate(prompt_messages_batch):
        refreshed_summary = refreshed_summaries[prompt_idx]
        if refreshed_summary:
            stored_memory_texts[prompt_idx] = _clip_text_to_token_budget(
                refreshed_summary,
                tokenizer=tokenizer,
                max_tokens=config.memory_max_tokens,
            )

        should_inject = (
            hard_prompt_mask[prompt_idx]
            and bool(stored_memory_texts[prompt_idx])
            and random.random() >= config.memory_dropout_rate
        )
        if (
            hard_prompt_mask[prompt_idx]
            and bool(stored_memory_texts[prompt_idx])
            and not should_inject
        ):
            dropped_prompt_count += 1

        if should_inject:
            injected_memory_texts.append(stored_memory_texts[prompt_idx])
            conditioned_messages_batch.append(
                build_conditioned_prompt(
                    prompt_messages,
                    wrap_in_framing(stored_memory_texts[prompt_idx]),
                )
            )
            injected_prompt_count += 1
        else:
            conditioned_messages_batch.append(prompt_messages)

    conditioned_batch = _generate_policy_rollouts(
        model=model,
        tokenizer=tokenizer,
        prompt_messages_batch=conditioned_messages_batch,
        answers=answers,
        config=config,
        label="retro_conditioned",
    )

    rewards_by_prompt = _group_rewards_by_prompt(
        conditioned_batch.rewards,
        conditioned_batch.group_ids,
        num_prompts=num_prompts,
    )

    refreshed_prompt_count = 0
    created_count = 0
    updated_count = 0
    changed_count = 0
    decayed_count = 0
    retired_count = 0
    summary_stability_total = 0.0

    for prompt_idx, prompt_id in enumerate(prompt_ids):
        observed_solve_rate = (
            sum(rewards_by_prompt[prompt_idx]) / len(rewards_by_prompt[prompt_idx])
            if rewards_by_prompt[prompt_idx]
            else 0.0
        )
        updated_solve_rate = update_solve_rate_ema(
            prompt_solve_emas.get(prompt_id),
            observed_solve_rate,
            config.memory_solve_rate_ema_alpha,
        )
        prompt_solve_emas[prompt_id] = updated_solve_rate

        existing_record = existing_records[prompt_idx]
        stored_summary = stored_memory_texts[prompt_idx]

        if stored_summary:
            refreshed_prompt_count += 1
            if existing_record is None:
                stability_score = 0.5
                created_count += 1
            else:
                overlap = compute_failure_mode_overlap(
                    existing_record.failure_modes,
                    extract_failure_modes(
                        stored_summary,
                        max_modes=config.memory_failure_mode_count,
                    ),
                )
                stability_score = min(
                    max((existing_record.stability_score + overlap) / 2.0, 0.0),
                    1.0,
                )
                updated_count += 1
                if existing_record.summary_text.strip() != stored_summary.strip():
                    changed_count += 1
            summary_stability_total += stability_score

            refreshed_record = build_memory_record(
                prompt_id=prompt_id,
                summary_text=stored_summary,
                solve_rate_ema=updated_solve_rate,
                last_updated_step=global_step,
                age=0,
                stability_score=stability_score,
                max_failure_modes=config.memory_failure_mode_count,
            )
            if should_retire_record(
                refreshed_record,
                hard_prompt_threshold=config.memory_hard_prompt_threshold,
                max_age=config.memory_ttl,
            ):
                memory_store.pop(prompt_id, None)
                if existing_record is not None:
                    retired_count += 1
            else:
                memory_store[prompt_id] = refreshed_record
            continue

        if existing_record is None:
            continue

        decayed_record = decay_memory_record(
            existing_record,
            solve_rate_ema=updated_solve_rate,
            decay_factor=config.memory_decay_factor,
        )
        if should_retire_record(
            decayed_record,
            hard_prompt_threshold=config.memory_hard_prompt_threshold,
            max_age=config.memory_ttl,
        ):
            memory_store.pop(prompt_id, None)
            retired_count += 1
        else:
            memory_store[prompt_id] = decayed_record
            decayed_count += 1

    injected_memory_lengths = []
    refreshed_memory_lengths = []
    available_memory_texts = [summary for summary in stored_memory_texts if summary]
    if available_memory_texts:
        refreshed_memory_lengths = [
            len(token_ids)
            for token_ids in tokenizer(
                available_memory_texts,
                add_special_tokens=False,
            ).input_ids
        ]
    if injected_memory_texts:
        injected_memory_lengths = [
            len(token_ids)
            for token_ids in tokenizer(
                injected_memory_texts,
                add_special_tokens=False,
            ).input_ids
        ]

    conditioned_batch.metrics.update(scout_batch.metrics)
    conditioned_batch.metrics["retro_failed_rollout_count"] = float(
        sum(len(failed_rollouts) for failed_rollouts in failed_rollouts_by_prompt)
    )
    conditioned_batch.metrics["retro_summary_generate_s"] = summary_generate_s
    conditioned_batch.metrics["retro_summary_count"] = float(
        sum(1 for summary in current_summaries if summary)
    )
    conditioned_batch.metrics["retro_summary_tokens"] = float(sum(refreshed_memory_lengths))
    conditioned_batch.metrics["retro_summary_tokens_max"] = (
        float(max(refreshed_memory_lengths)) if refreshed_memory_lengths else 0.0
    )
    conditioned_batch.metrics["retro_summary_prompt_count"] = float(injected_prompt_count)
    conditioned_batch.metrics["retro_prompt_count"] = float(num_prompts)

    conditioned_batch.metrics["long_memory_existing_prompt_count"] = float(existing_prompt_count)
    conditioned_batch.metrics["long_memory_current_failure_prompt_count"] = float(
        current_failure_prompt_count
    )
    conditioned_batch.metrics["long_memory_refreshed_prompt_count"] = float(
        refreshed_prompt_count
    )
    conditioned_batch.metrics["long_memory_merged_prompt_count"] = float(merged_prompt_count)
    conditioned_batch.metrics["long_memory_injected_prompt_count"] = float(
        injected_prompt_count
    )
    conditioned_batch.metrics["long_memory_dropped_prompt_count"] = float(dropped_prompt_count)
    conditioned_batch.metrics["long_memory_stale_prompt_count"] = float(stale_prompt_count)
    conditioned_batch.metrics["long_memory_created_count"] = float(created_count)
    conditioned_batch.metrics["long_memory_updated_count"] = float(updated_count)
    conditioned_batch.metrics["long_memory_changed_count"] = float(changed_count)
    conditioned_batch.metrics["long_memory_decayed_count"] = float(decayed_count)
    conditioned_batch.metrics["long_memory_retired_count"] = float(retired_count)
    conditioned_batch.metrics["long_memory_refresh_generate_s"] = refresh_generate_s
    conditioned_batch.metrics["long_memory_tokens"] = float(sum(injected_memory_lengths))
    conditioned_batch.metrics["long_memory_tokens_max"] = (
        float(max(injected_memory_lengths)) if injected_memory_lengths else 0.0
    )
    conditioned_batch.metrics["long_memory_available_tokens"] = float(
        sum(refreshed_memory_lengths)
    )
    conditioned_batch.metrics["long_memory_stability_total"] = float(
        summary_stability_total
    )
    conditioned_batch.metrics["long_memory_prompt_count"] = float(num_prompts)
    return conditioned_batch
