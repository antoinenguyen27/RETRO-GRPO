"""One-time setup for long-memory RETRO-GRPO assets."""

import os

import modal

from src.config import EVAL_DATA_DIR, MODEL_DIR, STAGE1_CONFIG, TRAIN_DATA_DIR


app = modal.App("long-retro-grpo-setup")

model_volume = modal.Volume.from_name("retro-grpo-models", create_if_missing=True)
data_volume = modal.Volume.from_name("retro-grpo-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def setup():
    from huggingface_hub import snapshot_download

    from src.data import load_and_prepare_eval_data, load_and_prepare_training_data

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already present at {MODEL_DIR}, skipping download.")
    else:
        print(f"Downloading {STAGE1_CONFIG.model_name} ...")
        snapshot_download(repo_id=STAGE1_CONFIG.model_name, local_dir=MODEL_DIR)
        model_volume.commit()

    if os.path.exists(TRAIN_DATA_DIR) and os.listdir(TRAIN_DATA_DIR):
        print(f"Training data already present at {TRAIN_DATA_DIR}, skipping.")
    else:
        load_and_prepare_training_data(save_path=TRAIN_DATA_DIR)
        data_volume.commit()

    if os.path.exists(EVAL_DATA_DIR) and os.listdir(EVAL_DATA_DIR):
        print(f"Eval data already present at {EVAL_DATA_DIR}, skipping.")
    else:
        load_and_prepare_eval_data(save_path=EVAL_DATA_DIR)
        data_volume.commit()

    for path in [MODEL_DIR, TRAIN_DATA_DIR, EVAL_DATA_DIR]:
        exists = os.path.exists(path) and bool(os.listdir(path))
        print(f"{path}: {'OK' if exists else 'MISSING'}")


@app.local_entrypoint()
def main():
    setup.remote()
