"""Modal app for RETRO-GRPO Stage 1 training.

Entry points:
    modal run train.py --mode baseline
    modal run train.py --mode retro
    modal run train.py --mode smoke_baseline --steps 10
    modal run train.py --mode smoke_retro --steps 10
"""

import modal

from .config import STAGE1_CONFIG, Stage1Config

app = modal.App("retro-grpo-train")

model_volume = modal.Volume.from_name("retro-grpo-models", create_if_missing=True)
data_volume = modal.Volume.from_name("retro-grpo-data", create_if_missing=True)

VOLUME_MOUNTS = {"/models": model_volume, "/data": data_volume}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", remote_path="/root/src")
)


def _clone_config() -> Stage1Config:
    return Stage1Config(**STAGE1_CONFIG.to_dict())


def _build_trainer(mode: str, run_name: str, output_dir: str):
    from datasets import load_from_disk

    from .config import MODEL_DIR, TRAIN_DATA_DIR
    from .modeling import load_trainable_policy
    from .trainer import BaselineTrainer, RetroTrainer

    config = _clone_config()
    dataset = load_from_disk(TRAIN_DATA_DIR)
    model, tokenizer = load_trainable_policy(
        config=config, model_name_or_path=MODEL_DIR
    )

    trainer_cls = BaselineTrainer if mode == "baseline" else RetroTrainer
    return trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=config,
        run_name=run_name,
        output_dir=output_dir,
    )


def _save_final_checkpoint(trainer, path: str) -> None:
    from .modeling import save_adapter_checkpoint

    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        save_adapter_checkpoint(
            model=trainer.accelerator.unwrap_model(trainer.model),
            tokenizer=trainer.tokenizer,
            output_dir=path,
            config=trainer.config,
            metadata={"final_optimizer_step": trainer.global_step},
        )


@app.function(
    gpu=STAGE1_CONFIG.gpu,
    image=image,
    volumes=VOLUME_MOUNTS,
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_baseline():
    trainer = _build_trainer(
        mode="baseline",
        run_name="baseline_grpo",
        output_dir="/models/baseline_checkpoints",
    )
    trainer.train()
    _save_final_checkpoint(trainer, "/models/baseline_final")
    model_volume.commit()


@app.function(
    gpu=STAGE1_CONFIG.gpu,
    image=image,
    volumes=VOLUME_MOUNTS,
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_retro():
    trainer = _build_trainer(
        mode="retro",
        run_name="retro_grpo",
        output_dir="/models/retro_checkpoints",
    )
    trainer.train()
    _save_final_checkpoint(trainer, "/models/retro_final")
    model_volume.commit()


@app.function(
    gpu=STAGE1_CONFIG.gpu,
    image=image,
    volumes=VOLUME_MOUNTS,
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def smoke_test(mode: str = "baseline", steps: int = 10):
    from datasets import load_from_disk

    from .config import MODEL_DIR, TRAIN_DATA_DIR
    from .modeling import load_trainable_policy
    from .trainer import BaselineTrainer, RetroTrainer

    config = _clone_config()
    config.num_train_epochs = 1
    config.save_steps = steps + 1

    dataset = load_from_disk(TRAIN_DATA_DIR)
    model, tokenizer = load_trainable_policy(
        config=config, model_name_or_path=MODEL_DIR
    )
    trainer_cls = BaselineTrainer if mode == "baseline" else RetroTrainer
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=config,
        run_name=f"smoke_{mode}",
        output_dir=f"/models/smoke_{mode}",
    )
    trainer.train(max_steps=steps)


@app.local_entrypoint()
def main(mode: str = "baseline", steps: int = -1):
    if mode == "baseline":
        train_baseline.remote()
    elif mode == "retro":
        train_retro.remote()
    elif mode.startswith("smoke"):
        actual_mode = mode.replace("smoke_", "")
        smoke_test.remote(mode=actual_mode, steps=steps if steps > 0 else 10)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use baseline, retro, smoke_baseline, or smoke_retro."
        )
