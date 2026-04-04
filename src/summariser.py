import random

import torch

from src.config import SYSTEM_PROMPT

SUMMARISER_TEMPLATE = """\
Summarise the approaches taken across these failed solution attempts in \
3-5 sentences. Describe what was tried and how the attempts ended up. \
Do not evaluate whether any approach was correct or incorrect. Do not \
suggest alternatives.

## Problem
{problem_text}

## Failed Attempts
{failed_attempts}

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


def build_summariser_prompt(question: str, failed_rollouts: list[str]) -> str:
    """Build the summariser input from a question and its failed rollout texts."""
    attempts_text = ""
    for i, rollout in enumerate(failed_rollouts, 1):
        attempts_text += f"\n### Attempt {i}\n{rollout.strip()}\n"
    return SUMMARISER_TEMPLATE.format(
        problem_text=question,
        failed_attempts=attempts_text,
    )


@torch.no_grad()
def generate_summary(
    model,
    tokenizer,
    question: str,
    failed_rollouts: list[str],
    max_new_tokens: int = 256,
) -> str:
    """Generate a descriptive failure summary using the training policy.

    Uses greedy decoding for deterministic, concise summaries.
    """
    prompt_text = build_summariser_prompt(question, failed_rollouts)

    messages = [
        {"role": "system", "content": "You are a concise summariser."},
        {"role": "user", "content": prompt_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        temperature=1.0,
    )
    summary_ids = outputs[0, input_ids.shape[1] :]
    summary = tokenizer.decode(summary_ids, skip_special_tokens=True).strip()
    return summary


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
