from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PackedRolloutTensors:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: torch.Tensor


@dataclass(slots=True)
class LossNormalizer:
    loss_type: str
    num_sequences: int
    total_active_tokens: int
    max_completion_length: int


LOGSUMEXP_CHUNK_SIZE = 4096


def pack_rollout_sequences(
    sequences: list[list[int]],
    prompt_lengths: list[int],
    pad_token_id: int,
    device: torch.device,
) -> PackedRolloutTensors:
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full(
        (len(sequences), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)

    for idx, seq in enumerate(sequences):
        seq_len = len(seq)
        start = max_len - seq_len
        # Left-pad packed sequences so all active suffixes align. This lets us
        # restrict logits to the completion window across the whole batch.
        input_ids[idx, start:] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[idx, start:] = 1

    return PackedRolloutTensors(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
    )


def compute_completion_logprobs(
    model,
    batch: PackedRolloutTensors,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_lengths = batch.attention_mask.sum(dim=1)
    completion_lengths = sequence_lengths - batch.prompt_lengths
    logits_to_keep = int(completion_lengths.max().item()) + 1

    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
        logits_to_keep=max(logits_to_keep, 2),
    )
    logits = outputs.logits[:, :-1, :]
    target_ids = batch.input_ids[:, -logits.shape[1] :]
    selected_logits = logits.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1).float()

    log_norm = None
    for start in range(0, logits.shape[-1], LOGSUMEXP_CHUNK_SIZE):
        chunk_log_norm = torch.logsumexp(
            logits[..., start : start + LOGSUMEXP_CHUNK_SIZE].float(), dim=-1
        )
        if log_norm is None:
            log_norm = chunk_log_norm
        else:
            log_norm = torch.logaddexp(log_norm, chunk_log_norm)

    token_logprobs = selected_logits - log_norm

    token_positions = torch.arange(1, batch.input_ids.shape[1], device=batch.input_ids.device)
    left_padding = batch.input_ids.shape[1] - sequence_lengths
    completion_starts = left_padding + batch.prompt_lengths
    full_completion_mask = (
        batch.attention_mask[:, 1:].bool()
        & (token_positions.unsqueeze(0) >= completion_starts.unsqueeze(1))
    )
    completion_mask = full_completion_mask[:, -logits.shape[1] :]

    token_logprobs = torch.where(
        completion_mask, token_logprobs, torch.zeros_like(token_logprobs)
    )
    return token_logprobs, completion_mask


def compute_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    scale_rewards: str = "group",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    advantages = torch.zeros_like(rewards)
    unique_groups = torch.unique(group_ids, sorted=True)
    zero_std_groups = 0

    if scale_rewards == "batch":
        batch_std = rewards.std(unbiased=False)
        if batch_std < eps:
            return advantages, {
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std(unbiased=False).item(),
                "zero_std_fraction": 1.0,
            }
    else:
        batch_std = None

    for group_id in unique_groups:
        mask = group_ids == group_id
        group_rewards = rewards[mask]
        centered = group_rewards - group_rewards.mean()

        if scale_rewards == "none":
            advantages[mask] = centered
            continue

        if scale_rewards == "group":
            denom = group_rewards.std(unbiased=False)
            if denom < eps:
                zero_std_groups += 1
                advantages[mask] = 0.0
            else:
                advantages[mask] = centered / denom
            continue

        if scale_rewards == "batch":
            advantages[mask] = centered / batch_std
            continue

        raise ValueError(f"Unsupported reward scaling mode: {scale_rewards}")

    zero_std_fraction = (
        zero_std_groups / len(unique_groups) if len(unique_groups) > 0 else 0.0
    )
    return advantages, {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std(unbiased=False).item(),
        "zero_std_fraction": zero_std_fraction,
    }


def build_loss_normalizer(
    completion_masks: list[torch.Tensor],
    loss_type: str,
    max_completion_length: int,
) -> LossNormalizer:
    total_active_tokens = sum(int(mask.sum().item()) for mask in completion_masks)
    num_sequences = sum(int(mask.shape[0]) for mask in completion_masks)
    return LossNormalizer(
        loss_type=loss_type,
        num_sequences=max(num_sequences, 1),
        total_active_tokens=max(total_active_tokens, 1),
        max_completion_length=max_completion_length,
    )


def compute_policy_objective(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float | None = 0.2,
    epsilon_high: float | None = None,
    ref_logprobs: torch.Tensor | None = None,
    beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    advantages = advantages.float().unsqueeze(1)
    current_logprobs = current_logprobs.float()
    old_logprobs = old_logprobs.float()
    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    clipped_ratio = ratio
    if epsilon is not None:
        upper = 1.0 + (epsilon_high if epsilon_high is not None else epsilon)
        clipped_ratio = ratio.clamp(1.0 - epsilon, upper)
        unclipped = ratio * advantages
        clipped = clipped_ratio * advantages
        policy_objective = torch.minimum(unclipped, clipped)
    else:
        policy_objective = ratio * advantages

    metrics = {
        "ratio_mean": ratio[completion_mask].mean().item()
        if completion_mask.any()
        else 1.0,
        "clip_fraction": (
            (ratio[completion_mask] != clipped_ratio[completion_mask])
            .float()
            .mean()
            .item()
            if completion_mask.any()
            else 0.0
        ),
    }

    if ref_logprobs is not None and beta > 0.0:
        log_ref_ratio = ref_logprobs.float() - current_logprobs
        per_token_kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
        policy_objective = policy_objective - beta * per_token_kl
        metrics["kl_mean"] = (
            per_token_kl[completion_mask].mean().item() if completion_mask.any() else 0.0
        )
    else:
        metrics["kl_mean"] = 0.0

    return torch.where(completion_mask, policy_objective, torch.zeros_like(policy_objective)), metrics


def reduce_policy_loss(
    per_token_objective: torch.Tensor,
    completion_mask: torch.Tensor,
    normalizer: LossNormalizer,
) -> torch.Tensor:
    token_counts = completion_mask.sum(dim=1).clamp_min(1)

    if normalizer.loss_type == "grpo":
        sequence_objective = per_token_objective.sum(dim=1) / token_counts
        return -(sequence_objective.sum() / normalizer.num_sequences)

    if normalizer.loss_type == "dapo":
        return -(per_token_objective.sum() / normalizer.total_active_tokens)

    if normalizer.loss_type == "dr_grpo":
        denom = normalizer.num_sequences * normalizer.max_completion_length
        return -(per_token_objective.sum() / max(denom, 1))

    raise ValueError(f"Unsupported loss reduction mode: {normalizer.loss_type}")
