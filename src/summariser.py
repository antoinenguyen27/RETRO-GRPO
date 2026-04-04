import random
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from src.token_utils import truncate_from_left


SUMMARISER_SYSTEM_PROMPT = "You are a careful summariser."

ROLLOUT_SUMMARISER_TEMPLATE = """\
Summarise the approach taken in this failed solution attempt in \
5-8 sentences. Describe what was tried and how the attempt ended up. \
Do not evaluate whether the approach was correct or incorrect. Do not \
suggest alternatives.

## Problem
{problem_text}

## Failed Attempt
{failed_attempt}

Summary of approach tried:"""

AGGREGATE_SUMMARISER_TEMPLATE = """\
Summarise the approaches taken across these failed solution-attempt \
summaries in 8-12 sentences. Describe what was tried and how the attempts \
ended up. Do not evaluate whether any approach was correct or incorrect. \
Do not suggest alternatives.

## Problem
{problem_text}

## Failed Attempt Summaries
{failed_attempt_summaries}

Summary of approaches tried:"""

FRAMING_VARIANTS = [
    "Previous attempts on this problem did not succeed. {summary}",
    (
        "Note: earlier solution attempts for this problem were tried. "
        "{summary} None of these succeeded."
    ),
    (
        "The following approaches were tried on this problem and did not "
        "produce a correct result. {summary}"
    ),
    (
        "Prior approaches to this problem: {summary} "
        "These did not lead to a successful outcome."
    ),
]


@dataclass(slots=True)
class FailureSummaryRequest:
    question: str
    failed_rollouts: list[str]


def build_rollout_summary_prompt(question: str, failed_rollout: str) -> str:
    """Build the first-pass summariser input for one failed rollout."""
    return ROLLOUT_SUMMARISER_TEMPLATE.format(
        problem_text=question,
        failed_attempt=failed_rollout.strip(),
    )


def build_aggregate_summary_prompt(question: str, rollout_summaries: list[str]) -> str:
    """Build the aggregate summariser input from rollout-level summaries."""
    summaries_text = ""
    for i, summary in enumerate(rollout_summaries, 1):
        summaries_text += f"\n### Attempt {i}\n{summary.strip()}\n"
    return AGGREGATE_SUMMARISER_TEMPLATE.format(
        problem_text=question,
        failed_attempt_summaries=summaries_text,
    )


def _adapter_disabled_context(model, summariser_mode: str):
    if summariser_mode == "frozen_base" and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


@torch.no_grad()
def _generate_batch(
    model,
    tokenizer,
    messages_batch: list[list[dict]],
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    max_seq_length: int,
) -> list[str]:
    """Generate outputs for a batch of chat prompts."""
    max_prompt_tokens = max(max_seq_length - max_new_tokens, 1)
    prompt_token_lists = [
        truncate_from_left(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ),
            max_prompt_tokens,
        )
        for messages in messages_batch
    ]
    model_inputs = tokenizer.pad(
        {"input_ids": prompt_token_lists},
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[:, model_inputs["input_ids"].shape[1] :]
    return [
        text.strip()
        for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    ]


@torch.no_grad()
def generate_failure_summaries(
    model,
    tokenizer,
    requests: list[FailureSummaryRequest],
    rollout_summary_max_new_tokens: int = 384,
    aggregate_summary_max_new_tokens: int = 512,
    enable_thinking: bool = False,
    do_sample: bool = False,
    summariser_mode: str = "training_policy",
    max_seq_length: int = 2048,
) -> list[str | None]:
    """Generate aggregate descriptive summaries for multiple prompts in batch."""
    if not requests:
        return []

    rollout_messages_batch: list[list[dict]] = []
    owners: list[int] = []
    for request_idx, request in enumerate(requests):
        for rollout in request.failed_rollouts:
            rollout_messages_batch.append(
                [
                    {"role": "system", "content": SUMMARISER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_rollout_summary_prompt(
                            request.question, rollout
                        ),
                    },
                ]
            )
            owners.append(request_idx)

    rollout_summaries_by_request: list[list[str]] = [[] for _ in requests]
    if rollout_messages_batch:
        with _adapter_disabled_context(model, summariser_mode):
            rollout_summaries = _generate_batch(
                model,
                tokenizer,
                rollout_messages_batch,
                max_new_tokens=rollout_summary_max_new_tokens,
                enable_thinking=enable_thinking,
                do_sample=do_sample,
                max_seq_length=max_seq_length,
            )
        for owner, summary in zip(owners, rollout_summaries):
            rollout_summaries_by_request[owner].append(summary)

    aggregate_messages_batch: list[list[dict]] = []
    aggregate_owners: list[int] = []
    for request_idx, request in enumerate(requests):
        if not rollout_summaries_by_request[request_idx]:
            continue
        aggregate_messages_batch.append(
            [
                {"role": "system", "content": SUMMARISER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_aggregate_summary_prompt(
                        request.question, rollout_summaries_by_request[request_idx]
                    ),
                },
            ]
        )
        aggregate_owners.append(request_idx)

    final_summaries: list[str | None] = [None] * len(requests)
    if aggregate_messages_batch:
        with _adapter_disabled_context(model, summariser_mode):
            aggregate_outputs = _generate_batch(
                model,
                tokenizer,
                aggregate_messages_batch,
                max_new_tokens=aggregate_summary_max_new_tokens,
                enable_thinking=enable_thinking,
                do_sample=do_sample,
                max_seq_length=max_seq_length,
            )
        for owner, summary in zip(aggregate_owners, aggregate_outputs):
            final_summaries[owner] = summary

    return final_summaries


@torch.no_grad()
def generate_summary(
    model,
    tokenizer,
    question: str,
    failed_rollouts: list[str],
    rollout_summary_max_new_tokens: int = 384,
    aggregate_summary_max_new_tokens: int = 512,
    enable_thinking: bool = False,
    do_sample: bool = False,
    summariser_mode: str = "training_policy",
    max_seq_length: int = 2048,
) -> str:
    """Generate one descriptive failure summary for one prompt."""
    summaries = generate_failure_summaries(
        model=model,
        tokenizer=tokenizer,
        requests=[FailureSummaryRequest(question=question, failed_rollouts=failed_rollouts)],
        rollout_summary_max_new_tokens=rollout_summary_max_new_tokens,
        aggregate_summary_max_new_tokens=aggregate_summary_max_new_tokens,
        enable_thinking=enable_thinking,
        do_sample=do_sample,
        summariser_mode=summariser_mode,
        max_seq_length=max_seq_length,
    )
    return summaries[0] or ""


def wrap_in_framing(summary: str) -> str:
    """Wrap a summary in a randomly selected framing variant."""
    variant = random.choice(FRAMING_VARIANTS)
    return variant.format(summary=summary)


def build_conditioned_prompt(
    original_messages: list[dict], failure_context: str
) -> list[dict]:
    """Prepend failure context to the user message in the prompt."""
    conditioned = []
    for msg in original_messages:
        if msg["role"] == "user":
            conditioned.append(
                {
                    "role": "user",
                    "content": f"[{failure_context}]\n\n{msg['content']}",
                }
            )
        else:
            conditioned.append(msg)
    return conditioned
