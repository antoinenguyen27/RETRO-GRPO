"""Modal app for MATH-500 evaluation.

Usage:
    modal run evaluate.py -- --checkpoint /models/baseline_final
    modal run evaluate.py -- --checkpoint /models/retro_final
"""

import json
import os

import modal

app = modal.App("retro-grpo-eval")

model_volume = modal.Volume.from_name("retro-grpo-models", create_if_missing=True)
data_volume = modal.Volume.from_name("retro-grpo-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "unsloth",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "wandb",
        "sympy",
    )
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    gpu="L4",
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def evaluate(checkpoint: str = "/models/baseline_final", run_name: str = ""):
    """Evaluate a checkpoint on MATH-500.

    Args:
        checkpoint: Path to the saved model checkpoint on the model volume.
        run_name: W&B run name for logging. Defaults to checkpoint dirname.
    """
    import sys

    sys.path.insert(0, "/root")

    from collections import defaultdict

    import torch
    import wandb
    from datasets import load_from_disk
    from unsloth import FastLanguageModel

    from src.reward import extract_boxed_answer, is_equivalent, normalize_answer

    # Determine run name from checkpoint path if not provided
    if not run_name:
        run_name = f"eval_{os.path.basename(checkpoint)}"

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project="retro-grpo-poc", name=run_name, job_type="eval")

    # Load model
    print(f"Loading model from {checkpoint} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=2048,
        dtype="bfloat16",
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load eval data
    print("Loading MATH-500 ...")
    dataset = load_from_disk("/data/math500")
    print(f"  {len(dataset)} problems loaded")

    # Evaluate
    results = []
    correct_total = 0
    correct_by_level = defaultdict(int)
    count_by_level = defaultdict(int)
    correct_by_subject = defaultdict(int)
    count_by_subject = defaultdict(int)

    print("Running evaluation ...")
    for i, example in enumerate(dataset):
        prompt_messages = example["prompt"]
        gt_answer = example["answer"]
        level = example["level"]
        subject = example["subject"]

        # Tokenize
        input_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate (greedy)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )
        completion_ids = outputs[0, input_ids.shape[1] :]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Verify
        pred = extract_boxed_answer(completion_text)
        is_correct = pred is not None and is_equivalent(
            normalize_answer(pred), normalize_answer(gt_answer)
        )

        results.append(
            {
                "idx": i,
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

        if (i + 1) % 50 == 0:
            print(
                f"  [{i + 1}/{len(dataset)}] Running accuracy: {correct_total / (i + 1):.3f}"
            )

    # Compute metrics
    total = len(dataset)
    overall_acc = correct_total / total

    # Hard slice: levels 4-5
    hard_correct = correct_by_level.get(4, 0) + correct_by_level.get(5, 0)
    hard_total = count_by_level.get(4, 0) + count_by_level.get(5, 0)
    hard_acc = hard_correct / hard_total if hard_total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"MATH-500 Evaluation Results: {run_name}")
    print("=" * 60)
    print(f"  Overall accuracy:    {correct_total}/{total} = {overall_acc:.4f}")
    print(f"  Hard slice (L4-L5):  {hard_correct}/{hard_total} = {hard_acc:.4f}")
    print()
    print("  By Level:")
    for level in sorted(count_by_level.keys()):
        c = correct_by_level[level]
        t = count_by_level[level]
        print(f"    Level {level}: {c}/{t} = {c / t:.3f}")
    print()
    print("  By Subject:")
    for subject in sorted(count_by_subject.keys()):
        c = correct_by_subject[subject]
        t = count_by_subject[subject]
        print(f"    {subject}: {c}/{t} = {c / t:.3f}")

    # Log to W&B
    wandb.log(
        {
            "eval/overall_accuracy": overall_acc,
            "eval/hard_slice_accuracy": hard_acc,
            "eval/total_correct": correct_total,
            "eval/total_problems": total,
        }
    )

    # Log per-level metrics
    for level in sorted(count_by_level.keys()):
        c = correct_by_level[level]
        t = count_by_level[level]
        wandb.log({f"eval/level_{level}_accuracy": c / t})

    # Log per-subject metrics
    for subject in sorted(count_by_subject.keys()):
        c = correct_by_subject[subject]
        t = count_by_subject[subject]
        wandb.log({f"eval/subject_{subject}_accuracy": c / t})

    # Save detailed results
    results_path = f"{checkpoint}_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    wandb.finish()
    model_volume.commit()

    return {"overall_accuracy": overall_acc, "hard_slice_accuracy": hard_acc}


@app.local_entrypoint()
def main(checkpoint: str = "/models/baseline_final", run_name: str = ""):
    result = evaluate.remote(checkpoint=checkpoint, run_name=run_name)
    print(f"\nResult: {result}")
