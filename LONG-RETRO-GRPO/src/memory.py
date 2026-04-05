from __future__ import annotations

import re
from dataclasses import dataclass


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class MemoryRecord:
    prompt_id: str
    summary_text: str
    failure_modes: list[str]
    solve_rate_ema: float
    last_updated_step: int
    age: int
    stability_score: float


def extract_failure_modes(summary_text: str, max_modes: int = 3) -> list[str]:
    """Extract a compact set of failure-mode strings from a narrative summary."""
    if not summary_text.strip():
        return []

    sentences = [
        sentence.strip(" -\n\t")
        for sentence in SENTENCE_SPLIT_RE.split(summary_text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return [summary_text.strip()]
    return sentences[:max(max_modes, 1)]


def update_solve_rate_ema(
    previous_value: float | None,
    observed_value: float,
    alpha: float,
) -> float:
    if previous_value is None:
        return observed_value
    return (1.0 - alpha) * previous_value + alpha * observed_value


def compute_failure_mode_overlap(old_modes: list[str], new_modes: list[str]) -> float:
    old_set = {mode.casefold() for mode in old_modes if mode.strip()}
    new_set = {mode.casefold() for mode in new_modes if mode.strip()}
    if not old_set and not new_set:
        return 1.0
    if not old_set or not new_set:
        return 0.0
    intersection = len(old_set & new_set)
    union = len(old_set | new_set)
    return intersection / union if union > 0 else 0.0


def build_memory_record(
    prompt_id: str,
    summary_text: str,
    solve_rate_ema: float,
    last_updated_step: int,
    age: int,
    stability_score: float,
    max_failure_modes: int,
) -> MemoryRecord:
    return MemoryRecord(
        prompt_id=prompt_id,
        summary_text=summary_text,
        failure_modes=extract_failure_modes(summary_text, max_modes=max_failure_modes),
        solve_rate_ema=solve_rate_ema,
        last_updated_step=last_updated_step,
        age=age,
        stability_score=stability_score,
    )


def decay_memory_record(
    record: MemoryRecord,
    solve_rate_ema: float,
    decay_factor: float,
) -> MemoryRecord:
    return MemoryRecord(
        prompt_id=record.prompt_id,
        summary_text=record.summary_text,
        failure_modes=record.failure_modes,
        solve_rate_ema=solve_rate_ema,
        last_updated_step=record.last_updated_step,
        age=record.age + 1,
        stability_score=max(min(record.stability_score * decay_factor, 1.0), 0.0),
    )


def should_retire_record(
    record: MemoryRecord,
    hard_prompt_threshold: float,
    max_age: int,
) -> bool:
    return (
        record.solve_rate_ema >= hard_prompt_threshold
        or record.age >= max_age
        or not record.summary_text.strip()
    )
