"""Custom rollout engines for baseline GRPO and RETRO-GRPO."""

from dataclasses import dataclass, field
from time import perf_counter

import torch

from src.config import Stage1Config
from src.reward import score_completion_texts
from src.summariser import (
    FailureSummaryRequest,
    build_conditioned_prompt,
    generate_failure_summaries,
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
    sequence_lengths = [prompt + completion for prompt, completion in zip(prompt_lengths, completion_lengths)]

    metrics = {
        f"{label}_correct": float(sum(rewards)),
        f"{label}_count": float(len(rewards)),
        f"{label}_truncated": float(sum(1 for item in truncated if item)),
        f"{label}_tokenize_s": tokenize_s,
        f"{label}_generate_s": generate_s,
        f"{label}_postprocess_s": postprocess_s,
        f"{label}_reward_s": reward_s,
        f"{label}_prompt_tokens": float(sum(prompt_token_counts)),
        f"{label}_prompt_tokens_max": float(max(prompt_token_counts)) if prompt_token_counts else 0.0,
        f"{label}_completion_tokens": float(sum(completion_lengths)),
        f"{label}_completion_tokens_max": float(max(completion_lengths)) if completion_lengths else 0.0,
        f"{label}_sequence_tokens": float(sum(sequence_lengths)),
        f"{label}_sequence_tokens_max": float(max(sequence_lengths)) if sequence_lengths else 0.0,
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
    """Generate the standard GRPO completion groups."""
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
    config: Stage1Config,
) -> RolloutBatch:
    """Generate RETRO scout rollouts, summaries, and conditioned rollouts."""
    scout_batch = _generate_policy_rollouts(
        model=model,
        tokenizer=tokenizer,
        prompt_messages_batch=prompt_messages_batch,
        answers=answers,
        config=config,
        label="retro_scout",
    )

    requests: list[FailureSummaryRequest] = []
    for prompt_idx, prompt_messages in enumerate(prompt_messages_batch):
        failed_rollouts = [
            completion_text
            for completion_text, reward, group_id in zip(
                scout_batch.completion_texts,
                scout_batch.rewards,
                scout_batch.group_ids,
            )
            if group_id == prompt_idx and reward == 0.0
        ]
        requests.append(
            FailureSummaryRequest(
                question=prompt_messages[-1]["content"],
                failed_rollouts=failed_rollouts,
            )
        )

    _synchronize_model_device(model)
    summary_start = perf_counter()
    summaries = generate_failure_summaries(
        model=model,
        tokenizer=tokenizer,
        requests=requests,
        rollout_summary_max_new_tokens=config.rollout_summary_max_new_tokens,
        aggregate_summary_max_new_tokens=config.aggregate_summary_max_new_tokens,
        enable_thinking=config.summary_enable_thinking,
        do_sample=config.summary_do_sample,
        summariser_mode=config.summariser_mode,
        max_seq_length=config.max_seq_length,
    )
    _synchronize_model_device(model)
    summary_generate_s = perf_counter() - summary_start

    non_empty_summaries = [summary for summary in summaries if summary]
    summary_token_lengths = []
    if non_empty_summaries:
        summary_token_lengths = [
            len(token_ids)
            for token_ids in tokenizer(
                non_empty_summaries, add_special_tokens=False
            ).input_ids
        ]

    conditioned_messages_batch: list[list[dict]] = []
    conditioned_prompt_count = 0
    for prompt_messages, summary in zip(prompt_messages_batch, summaries):
        if summary:
            conditioned_prompt_count += 1
            conditioned_messages_batch.append(
                build_conditioned_prompt(prompt_messages, wrap_in_framing(summary))
            )
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
    conditioned_batch.metrics.update(scout_batch.metrics)
    conditioned_batch.metrics["retro_failed_rollout_count"] = float(
        sum(len(request.failed_rollouts) for request in requests)
    )
    conditioned_batch.metrics["retro_summary_generate_s"] = summary_generate_s
    conditioned_batch.metrics["retro_summary_count"] = float(len(non_empty_summaries))
    conditioned_batch.metrics["retro_summary_tokens"] = float(sum(summary_token_lengths))
    conditioned_batch.metrics["retro_summary_tokens_max"] = (
        float(max(summary_token_lengths)) if summary_token_lengths else 0.0
    )
    conditioned_batch.metrics["retro_summary_prompt_count"] = float(
        conditioned_prompt_count
    )
    conditioned_batch.metrics["retro_prompt_count"] = float(len(prompt_messages_batch))
    return conditioned_batch
