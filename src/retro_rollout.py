"""Custom rollout engines for baseline GRPO and RETRO-GRPO."""

from dataclasses import dataclass, field

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

    repeated_answers = [answers[group_id] for group_id in group_ids]
    rewards = score_completion_texts(completion_texts, repeated_answers)
    completion_lengths = [len(ids) for ids in completion_ids]

    metrics = {
        f"{label}_correct": float(sum(rewards)),
        f"{label}_count": float(len(rewards)),
        f"{label}_truncated": float(sum(1 for item in truncated if item)),
        f"{label}_completion_tokens": float(sum(completion_lengths)),
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
    conditioned_batch.metrics["retro_summary_prompt_count"] = float(
        conditioned_prompt_count
    )
    conditioned_batch.metrics["retro_prompt_count"] = float(len(prompt_messages_batch))
    return conditioned_batch
