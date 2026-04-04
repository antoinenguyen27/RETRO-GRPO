"""One-time setup: download model weights and datasets to Modal volumes."""

import os

import modal

app = modal.App("retro-grpo-setup")

model_volume = modal.Volume.from_name("retro-grpo-models", create_if_missing=True)
data_volume = modal.Volume.from_name("retro-grpo-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "accelerate",
    )
)


@app.function(
    gpu="L4",
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def setup():
    """Download model and datasets to persistent volumes if not already present."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    model_dir = "/models/Qwen3.5-4B"
    train_data_dir = "/data/deepmath_hard_1200"
    eval_data_dir = "/data/math500"

    # --- Download model ---
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Model already present at {model_dir}, skipping download.")
    else:
        print("Downloading Qwen/Qwen3.5-4B ...")
        snapshot_download(
            repo_id="Qwen/Qwen3.5-4B",
            local_dir=model_dir,
        )
        print(f"Model saved to {model_dir}")
        model_volume.commit()

    # --- Download and process training data ---
    if os.path.exists(train_data_dir) and os.listdir(train_data_dir):
        print(f"Training data already present at {train_data_dir}, skipping.")
    else:
        print("Downloading and filtering DeepMath-103K ...")
        ds = load_dataset("zwhe99/DeepMath-103K", split="train")
        print(f"  Full dataset: {len(ds)} examples")

        # Sort by difficulty descending, take top 1200
        ds = ds.sort("difficulty", reverse=True)
        ds = ds.select(range(1200))
        print(f"  Hard-filtered subset: {len(ds)} examples")
        print(f"  Difficulty range: {ds[-1]['difficulty']:.2f} - {ds[0]['difficulty']:.2f}")

        # Format for GRPOTrainer (conversational format with final_answer)
        system_prompt = (
            "Solve the following math problem step by step. "
            "Provide your final answer within \\boxed{}."
        )

        def format_row(example):
            return {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["question"]},
                ],
                "final_answer": example["final_answer"],
            }

        ds = ds.map(format_row, remove_columns=ds.column_names)
        ds.save_to_disk(train_data_dir)
        print(f"Training data saved to {train_data_dir}")
        data_volume.commit()

    # --- Download and process eval data ---
    if os.path.exists(eval_data_dir) and os.listdir(eval_data_dir):
        print(f"Eval data already present at {eval_data_dir}, skipping.")
    else:
        print("Downloading MATH-500 ...")
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"  MATH-500: {len(ds)} examples")

        system_prompt = (
            "Solve the following math problem step by step. "
            "Provide your final answer within \\boxed{}."
        )

        def format_eval(example):
            return {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["problem"]},
                ],
                "answer": example["answer"],
                "level": example["level"],
                "subject": example["subject"],
            }

        ds = ds.map(format_eval, remove_columns=ds.column_names)
        ds.save_to_disk(eval_data_dir)
        print(f"Eval data saved to {eval_data_dir}")
        data_volume.commit()

    print("\nSetup complete. Volume contents:")
    for d in [model_dir, train_data_dir, eval_data_dir]:
        exists = os.path.exists(d) and os.listdir(d)
        print(f"  {d}: {'OK' if exists else 'MISSING'}")


@app.local_entrypoint()
def main():
    setup.remote()
