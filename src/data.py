from datasets import load_dataset, Dataset

from src.config import SYSTEM_PROMPT


def filter_hard_problems(dataset: Dataset, n: int = 1200) -> Dataset:
    """Sort DeepMath-103K by difficulty descending and take the top n problems."""
    sorted_ds = dataset.sort("difficulty", reverse=True)
    return sorted_ds.select(range(min(n, len(sorted_ds))))


def format_for_training(example: dict) -> dict:
    """Convert a DeepMath-103K row into conversational training format.

    Returns a dict with 'prompt' (list of message dicts) and 'final_answer'.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "final_answer": example["final_answer"],
    }


def format_for_eval(example: dict) -> dict:
    """Convert a MATH-500 row into evaluation format."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "answer": example["answer"],
        "level": example["level"],
        "subject": example["subject"],
    }


def load_and_prepare_training_data(save_path: str | None = None) -> Dataset:
    """Load DeepMath-103K, filter to hard problems, format for training."""
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = filter_hard_problems(ds, n=1200)
    ds = ds.map(format_for_training, remove_columns=ds.column_names)
    if save_path:
        ds.save_to_disk(save_path)
    return ds


def load_and_prepare_eval_data(save_path: str | None = None) -> Dataset:
    """Load MATH-500 and format for evaluation."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.map(format_for_eval, remove_columns=ds.column_names)
    if save_path:
        ds.save_to_disk(save_path)
    return ds
