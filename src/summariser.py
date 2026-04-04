import random

import torch

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


@torch.no_grad()
def _generate_batch(
    model,
    tokenizer,
    messages_batch: list[list[dict]],
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate deterministic text outputs for a batch of chat prompts."""
    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]
    model_inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = outputs[:, model_inputs["input_ids"].shape[1] :]
    return [
        text.strip()
        for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    ]


@torch.no_grad()
def generate_summary(
    model,
    tokenizer,
    question: str,
    failed_rollouts: list[str],
    rollout_summary_max_new_tokens: int = 384,
    aggregate_summary_max_new_tokens: int = 512,
) -> str:
    """Generate a descriptive failure summary via summarise-then-aggregate.

    Each failed rollout is first summarised individually in batch, then those
    rollout summaries are aggregated into one final descriptive narrative.
    """
    rollout_messages_batch = [
        [
            {"role": "system", "content": SUMMARISER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_rollout_summary_prompt(question, rollout),
            },
        ]
        for rollout in failed_rollouts
    ]
    rollout_summaries = _generate_batch(
        model,
        tokenizer,
        rollout_messages_batch,
        max_new_tokens=rollout_summary_max_new_tokens,
    )

    aggregate_messages = [
        {"role": "system", "content": SUMMARISER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_aggregate_summary_prompt(question, rollout_summaries),
        },
    ]
    return _generate_batch(
        model,
        tokenizer,
        [aggregate_messages],
        max_new_tokens=aggregate_summary_max_new_tokens,
    )[0]


def wrap_in_framing(summary: str) -> str:
    """Wrap a summary in a randomly selected framing variant."""
    variant = random.choice(FRAMING_VARIANTS)
    return variant.format(summary=summary)


def build_conditioned_prompt(
    original_messages: list[dict], failure_context: str
) -> list[dict]:
    """Prepend failure context to the user message in the prompt.

    Returns a new message list with the failure context block before the question.
    """
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
