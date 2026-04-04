import re

from sympy import simplify, sympify
from sympy.parsing.latex import parse_latex


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in the text.

    Handles nested braces by counting brace depth.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    # Walk forward from the opening brace, counting depth
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize a math answer string for comparison."""
    s = answer.strip()
    # Remove enclosing dollar signs
    s = s.strip("$")
    # Remove \\text{} wrappers
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_equivalent(predicted: str, ground_truth: str) -> bool:
    """Check if predicted and ground_truth answers are mathematically equivalent.

    Tries exact string match first, then numeric comparison, then sympy.
    """
    if predicted == ground_truth:
        return True

    # Try numeric comparison
    try:
        p_val = float(predicted.replace(",", ""))
        g_val = float(ground_truth.replace(",", ""))
        if abs(p_val - g_val) < 1e-6:
            return True
    except (ValueError, OverflowError):
        pass

    # Try sympy symbolic comparison
    try:
        p_expr = parse_latex(predicted)
        g_expr = parse_latex(ground_truth)
        if simplify(p_expr - g_expr) == 0:
            return True
    except Exception:
        pass

    # Try direct sympify (handles expressions like "1/2")
    try:
        p_expr = sympify(predicted)
        g_expr = sympify(ground_truth)
        if simplify(p_expr - g_expr) == 0:
            return True
    except Exception:
        pass

    return False


def score_completion_texts(
    completions: list[str], final_answers: list[str]
) -> list[float]:
    """Score completion texts against deterministic math answers."""
    rewards = []
    for completion, gt in zip(completions, final_answers):
        pred = extract_boxed_answer(completion)
        if pred is not None and is_equivalent(
            normalize_answer(pred), normalize_answer(gt)
        ):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions, final_answer, **kwargs) -> list[float]:
    """Compatibility wrapper for deterministic math answer verification.

    Args:
        completions: List of completion strings or message dicts.
        final_answer: List of ground-truth answer strings (from dataset column).

    Returns:
        List of floats: 1.0 for correct, 0.0 for incorrect.
    """
    rewards = []
    for completion, gt in zip(completions, final_answer):
        # Handle conversational format
        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = completion

        rewards.append(score_completion_texts([text], [gt])[0])
    return rewards
