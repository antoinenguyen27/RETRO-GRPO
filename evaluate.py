"""Modal app for MATH-500 evaluation."""

import json
import os
from collections import defaultdict

import modal

from src.config import STAGE1_CONFIG, Stage1Config
from src.token_utils import truncate_from_left


app = modal.App("retro-grpo-eval")

model_volume = modal.Volume.from_name("retro-grpo-models", create_if_missing=True)
data_volume = modal.Volume.from_name("retro-grpo-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", remote_path="/root/src")
)


def _clone_config() -> Stage1Config:
    return Stage1Config(**STAGE1_CONFIG.to_dict())


def _generate_batch(model, tokenizer, prompts: list[list[dict]], config: Stage1Config) -> list[str]:
    max_prompt_tokens = max(config.max_seq_length - config.max_completion_length, 1)
    prompt_token_lists = [
        truncate_from_left(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=config.eval_enable_thinking,
            ),
            max_prompt_tokens,
        )
        for messages in prompts
    ]
    model_inputs = tokenizer.pad(
        {"input_ids": prompt_token_lists},
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generation_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "max_new_tokens": config.max_completion_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.eval_do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": config.eval_temperature,
                "top_p": config.eval_top_p,
                "top_k": config.eval_top_k,
                "min_p": config.eval_min_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    outputs = model.generate(**generation_kwargs)
    generated_ids = outputs[:, model_inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


@app.function(
    gpu=STAGE1_CONFIG.gpu,
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def evaluate(
    checkpoint: str = "/models/baseline_final",
    run_name: str = "",
    batch_size: int = 8,
):
    import torch
    import wandb
    from datasets import load_from_disk

    from src.config import EVAL_DATA_DIR, MODEL_DIR
    from src.modeling import load_policy_for_inference
    from src.reward import extract_boxed_answer, is_equivalent, normalize_answer

    config = _clone_config()
    if not run_name:
        run_name = f"eval_{os.path.basename(checkpoint)}"

    torch.manual_seed(config.seed)
    wandb.init(project=config.wandb_project, name=run_name, job_type="eval")

    model, tokenizer = load_policy_for_inference(
        checkpoint_path=checkpoint,
        config=config,
        base_model_name=MODEL_DIR,
    )
    model.eval()
    model.to("cuda")

    dataset = load_from_disk(EVAL_DATA_DIR)
    results = []
    correct_total = 0
    correct_by_level = defaultdict(int)
    count_by_level = defaultdict(int)
    correct_by_subject = defaultdict(int)
    count_by_subject = defaultdict(int)

    for start in range(0, len(dataset), batch_size):
        rows = dataset[start : start + batch_size]
        prompts = rows["prompt"]
        answers = rows["answer"]
        levels = rows["level"]
        subjects = rows["subject"]

        completion_texts = _generate_batch(model, tokenizer, prompts, config)
        for offset, (completion_text, gt_answer, level, subject) in enumerate(
            zip(completion_texts, answers, levels, subjects)
        ):
            pred = extract_boxed_answer(completion_text)
            is_correct = pred is not None and is_equivalent(
                normalize_answer(pred), normalize_answer(gt_answer)
            )
            row_index = start + offset
            results.append(
                {
                    "idx": row_index,
                    "level": level,
                    "subject": subject,
                    "ground_truth": gt_answer,
                    "predicted": pred,
                    "correct": is_correct,
                }
            )

            if is_correct:
                correct_total += 1
                correct_by_level[level] += 1
                correct_by_subject[subject] += 1

            count_by_level[level] += 1
            count_by_subject[subject] += 1

    total = len(dataset)
    overall_acc = correct_total / total
    hard_correct = correct_by_level.get(4, 0) + correct_by_level.get(5, 0)
    hard_total = count_by_level.get(4, 0) + count_by_level.get(5, 0)
    hard_acc = hard_correct / hard_total if hard_total > 0 else 0.0

    wandb.log(
        {
            "eval/overall_accuracy": overall_acc,
            "eval/hard_slice_accuracy": hard_acc,
            "eval/total_correct": correct_total,
            "eval/total_problems": total,
        }
    )
    for level in sorted(count_by_level):
        wandb.log({f"eval/level_{level}_accuracy": correct_by_level[level] / count_by_level[level]})
    for subject in sorted(count_by_subject):
        wandb.log(
            {
                f"eval/subject_{subject}_accuracy": (
                    correct_by_subject[subject] / count_by_subject[subject]
                )
            }
        )
    wandb.finish()

    results_path = os.path.join(checkpoint, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    model_volume.commit()

    return {"overall_accuracy": overall_acc, "hard_slice_accuracy": hard_acc}


@app.local_entrypoint()
def main(
    checkpoint: str = "/models/baseline_final",
    run_name: str = "",
    batch_size: int = 8,
):
    result = evaluate.remote(checkpoint=checkpoint, run_name=run_name, batch_size=batch_size)
    print(f"\nResult: {result}")
