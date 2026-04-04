"""Modal app for RETRO-GRPO Stage 1 training.

Entry points:
    modal run train.py::train_baseline
    modal run train.py::train_retro
    modal run train.py::smoke_test --mode baseline --steps 10
"""

import os

import modal

app = modal.App("retro-grpo-train")

model_volume = modal.Volume.from_name("retro-grpo-models", create_if_missing=True)
data_volume = modal.Volume.from_name("retro-grpo-data", create_if_missing=True)

VOLUME_MOUNTS = {"/models": model_volume, "/data": data_volume}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "unsloth",
        "trl",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "wandb",
        "sympy",
    )
    .add_local_dir("src", remote_path="/root/src")
)


def _load_model_and_tokenizer(model_dir: str = "/models/Qwen3.5-4B"):
    """Load Qwen3.5-4B with bf16 LoRA via Unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        dtype="bfloat16",
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    # Ensure padding is set correctly for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def _get_grpo_config(
    run_name: str,
    output_dir: str,
    num_train_epochs: int = 3,
    max_steps: int = -1,
):
    """Build GRPOConfig for stage 1."""
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_generations=4,
        per_device_train_batch_size=3,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        max_completion_length=512,
        temperature=1.0,
        learning_rate=5e-6,
        beta=0.001,
        epsilon=0.2,
        # Logging
        report_to="wandb",
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        # Generation
        bf16=True,
        gradient_checkpointing=True,
        # Misc
        seed=42,
        dataloader_num_workers=0,
    )


# --------------------------------------------------------------------------- #
#  Baseline GRPO
# --------------------------------------------------------------------------- #


@app.function(
    gpu="L4",
    image=image,
    volumes=VOLUME_MOUNTS,
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_baseline():
    """Run standard GRPO training (baseline)."""
    import sys
    sys.path.insert(0, "/root")

    import wandb
    from datasets import load_from_disk
    from trl import GRPOTrainer

    from src.reward import accuracy_reward

    wandb.login(key=os.environ["WANDB_API_KEY"])

    print("Loading model ...")
    model, tokenizer = _load_model_and_tokenizer()

    print("Loading training data ...")
    dataset = load_from_disk("/data/deepmath_hard_1200")

    config = _get_grpo_config(
        run_name="baseline_grpo",
        output_dir="/models/baseline_checkpoints",
    )

    print("Starting baseline GRPO training ...")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save final model
    print("Saving final model ...")
    model.save_pretrained("/models/baseline_final")
    tokenizer.save_pretrained("/models/baseline_final")
    model_volume.commit()
    print("Baseline training complete.")


# --------------------------------------------------------------------------- #
#  RETRO-GRPO
# --------------------------------------------------------------------------- #


@app.function(
    gpu="L4",
    image=image,
    volumes=VOLUME_MOUNTS,
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_retro():
    """Run RETRO-GRPO training with two-phase failure-conditioned rollouts."""
    import sys
    sys.path.insert(0, "/root")

    import wandb
    from datasets import load_from_disk
    from trl import GRPOTrainer

    from src.retro_rollout import retro_grpo_rollout
    from src.reward import accuracy_reward

    wandb.login(key=os.environ["WANDB_API_KEY"])

    print("Loading model ...")
    model, tokenizer = _load_model_and_tokenizer()

    print("Loading training data ...")
    dataset = load_from_disk("/data/deepmath_hard_1200")

    config = _get_grpo_config(
        run_name="retro_grpo",
        output_dir="/models/retro_checkpoints",
    )

    # Build a lookup from question text → final_answer so the rollout
    # function can score scouts without needing access to the full dataset row.
    # The rollout_func only receives prompt messages, not extra columns.
    answer_lookup = {}
    for row in dataset:
        question = row["prompt"][-1]["content"]
        answer_lookup[question] = row["final_answer"]

    print("Starting RETRO-GRPO training ...")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_func=retro_grpo_rollout,
    )

    # Attach the answer lookup to the trainer so retro_grpo_rollout can use it
    trainer._retro_answer_lookup = answer_lookup

    trainer.train()

    # Save final model
    print("Saving final model ...")
    model.save_pretrained("/models/retro_final")
    tokenizer.save_pretrained("/models/retro_final")
    model_volume.commit()
    print("RETRO-GRPO training complete.")


# --------------------------------------------------------------------------- #
#  Smoke test (quick validation before full runs)
# --------------------------------------------------------------------------- #


@app.function(
    gpu="L4",
    image=image,
    volumes=VOLUME_MOUNTS,
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def smoke_test(mode: str = "baseline", steps: int = 10):
    """Run a short smoke test to validate the training pipeline.

    Args:
        mode: "baseline" or "retro"
        steps: Number of optimizer steps to run.
    """
    import sys
    sys.path.insert(0, "/root")

    import wandb
    from datasets import load_from_disk
    from trl import GRPOTrainer

    from src.reward import accuracy_reward

    wandb.login(key=os.environ["WANDB_API_KEY"])

    print(f"Smoke test: mode={mode}, steps={steps}")
    model, tokenizer = _load_model_and_tokenizer()
    dataset = load_from_disk("/data/deepmath_hard_1200")

    config = _get_grpo_config(
        run_name=f"smoke_{mode}",
        output_dir=f"/models/smoke_{mode}",
        max_steps=steps,
    )
    config.save_steps = steps + 1  # Don't save during smoke test

    kwargs = {}
    if mode == "retro":
        from src.retro_rollout import retro_grpo_rollout
        kwargs["rollout_func"] = retro_grpo_rollout

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
        **kwargs,
    )

    if mode == "retro":
        answer_lookup = {}
        for row in dataset:
            question = row["prompt"][-1]["content"]
            answer_lookup[question] = row["final_answer"]
        trainer._retro_answer_lookup = answer_lookup

    trainer.train()
    print(f"Smoke test ({mode}) passed: {steps} steps completed.")


@app.local_entrypoint()
def main(mode: str = "baseline", steps: int = -1):
    """CLI entry point.

    Usage:
        modal run train.py -- --mode baseline
        modal run train.py -- --mode retro
        modal run train.py -- --mode smoke_baseline --steps 10
        modal run train.py -- --mode smoke_retro --steps 10
    """
    if mode == "baseline":
        train_baseline.remote()
    elif mode == "retro":
        train_retro.remote()
    elif mode.startswith("smoke"):
        actual_mode = mode.replace("smoke_", "")
        smoke_test.remote(mode=actual_mode, steps=steps if steps > 0 else 10)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use baseline, retro, smoke_baseline, or smoke_retro.")
