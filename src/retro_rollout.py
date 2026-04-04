"""Custom rollout function for RETRO-GRPO two-phase pipeline.

Implements the in-step scout → summarise → conditioned-generation pipeline
via TRL's experimental rollout_func API.
"""

import torch
import torch.nn.functional as F

from src.reward import extract_boxed_answer, is_equivalent, normalize_answer
from src.summariser import (
    build_conditioned_prompt,
    generate_summary,
    wrap_in_framing,
)


def _compute_logprobs_batched(
    scores: tuple[torch.Tensor, ...], sequences: torch.Tensor, prompt_len: int
) -> list[list[float]]:
    """Compute per-token logprobs for a batch of sequences.

    Args:
        scores: Tuple of score tensors, one per generated step.
                Each shape: (batch_size, vocab_size).
        sequences: Full sequences tensor (batch_size, seq_len).
        prompt_len: Length of the prompt (shared across the batch).

    Returns:
        List of lists of logprobs, one inner list per sequence in the batch.
    """
    batch_size = sequences.shape[0]
    all_logprobs: list[list[float]] = [[] for _ in range(batch_size)]

    for step_idx, step_scores in enumerate(scores):
        step_log_probs = F.log_softmax(step_scores, dim=-1)  # (batch, vocab)
        token_pos = prompt_len + step_idx
        if token_pos >= sequences.shape[1]:
            break
        for b in range(batch_size):
            tid = sequences[b, token_pos].item()
            all_logprobs[b].append(step_log_probs[b, tid].item())

    return all_logprobs


def _score_completions(texts: list[str], ground_truth: str) -> list[float]:
    """Score completions against ground truth. Returns list of 0.0/1.0."""
    rewards = []
    for text in texts:
        pred = extract_boxed_answer(text)
        if pred is not None and is_equivalent(
            normalize_answer(pred), normalize_answer(ground_truth)
        ):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


@torch.no_grad()
def retro_grpo_rollout(prompts: list, trainer) -> dict:
    """Two-phase rollout for RETRO-GRPO.

    For each prompt:
      1. Generate N scout rollouts (unconditioned)
      2. Score scouts with deterministic verification
      3. Summarise each failed scout, then aggregate into one descriptive
         narrative
      4. Generate N conditioned rollouts (summary prepended to prompt)
      5. Return conditioned rollouts (prompt_ids, completion_ids, logprobs)

    The scouts are discarded after summarisation — only conditioned rollouts
    enter the GRPO gradient computation.

    Args:
        prompts: List of prompt message lists allocated to this process.
                 Each element is a list of message dicts (conversational format).
        trainer: The GRPOTrainer instance (provides model, tokenizer, config).

    Returns:
        Dict with keys: prompt_ids, completion_ids, logprobs, final_answer.
    """
    model = trainer.model
    tokenizer = trainer.processing_class
    num_gen = trainer.args.num_generations
    max_new_tokens = trainer.args.max_completion_length
    rollout_summary_max_new_tokens = getattr(
        trainer.args, "rollout_summary_max_new_tokens", 384
    )
    aggregate_summary_max_new_tokens = getattr(
        trainer.args, "aggregate_summary_max_new_tokens", 512
    )

    all_prompt_ids: list[list[int]] = []
    all_completion_ids: list[list[int]] = []
    all_logprobs: list[list[float]] = []
    all_final_answers: list[str] = []

    # RETRO-GRPO custom metrics accumulators
    total_scout_correct = 0
    total_scout_count = 0
    total_cond_correct = 0
    total_cond_count = 0

    # Answer lookup is attached to the trainer by train.py
    answer_lookup = trainer._retro_answer_lookup

    for prompt_messages in prompts:
        question = prompt_messages[-1]["content"]

        # --- Phase 1: Scout generation (unconditioned) ---
        scout_input_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        scout_outputs = model.generate(
            input_ids=scout_input_ids.expand(num_gen, -1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            return_dict_in_generate=True,
        )
        prompt_len = scout_input_ids.shape[1]
        scout_comp_ids = scout_outputs.sequences[:, prompt_len:]
        scout_texts = tokenizer.batch_decode(scout_comp_ids, skip_special_tokens=True)

        # --- Phase 2: Score scouts ---
        gt = answer_lookup[question]
        scout_rewards = _score_completions(scout_texts, gt)
        total_scout_correct += sum(scout_rewards)
        total_scout_count += len(scout_rewards)

        failed_scouts = [
            t for t, r in zip(scout_texts, scout_rewards) if r == 0.0
        ]

        # --- Phase 3: Summarise failures ---
        if failed_scouts:
            summary = generate_summary(
                model,
                tokenizer,
                question,
                failed_scouts,
                rollout_summary_max_new_tokens=rollout_summary_max_new_tokens,
                aggregate_summary_max_new_tokens=aggregate_summary_max_new_tokens,
            )
            failure_context = wrap_in_framing(summary)
            conditioned_messages = build_conditioned_prompt(
                prompt_messages, failure_context
            )
        else:
            # All scouts correct (rare on hard problems) — skip conditioning
            conditioned_messages = prompt_messages

        # --- Phase 4: Conditioned generation ---
        cond_input_ids = tokenizer.apply_chat_template(
            conditioned_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        cond_outputs = model.generate(
            input_ids=cond_input_ids.expand(num_gen, -1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        cond_prompt_len = cond_input_ids.shape[1]
        cond_sequences = cond_outputs.sequences

        # Compute logprobs for each conditioned completion
        batch_logprobs = _compute_logprobs_batched(
            cond_outputs.scores, cond_sequences, cond_prompt_len
        )

        # Collect conditioned results
        cond_prompt_ids_list = cond_input_ids[0].tolist()
        for seq_idx in range(num_gen):
            comp_ids = cond_sequences[seq_idx, cond_prompt_len:].tolist()
            # Trim trailing pad tokens
            if tokenizer.pad_token_id is not None:
                while comp_ids and comp_ids[-1] == tokenizer.pad_token_id:
                    comp_ids.pop()
            all_prompt_ids.append(cond_prompt_ids_list)
            all_completion_ids.append(comp_ids)
            all_logprobs.append(batch_logprobs[seq_idx][: len(comp_ids)])
            all_final_answers.append(gt)

        # Track conditioned solve rate
        cond_texts = tokenizer.batch_decode(
            cond_sequences[:, cond_prompt_len:], skip_special_tokens=True
        )
        cond_rewards = _score_completions(cond_texts, gt)
        total_cond_correct += sum(cond_rewards)
        total_cond_count += len(cond_rewards)

    # Log RETRO-specific metrics if W&B is available
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "retro/scout_solve_rate": (
                        total_scout_correct / total_scout_count
                        if total_scout_count > 0
                        else 0.0
                    ),
                    "retro/conditioned_solve_rate": (
                        total_cond_correct / total_cond_count
                        if total_cond_count > 0
                        else 0.0
                    ),
                    "retro/conditioning_rate": 1.0,  # Stage 1: always conditioned
                },
                commit=False,
            )
    except ImportError:
        pass

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "final_answer": all_final_answers,
    }
