import json
import os
from contextlib import nullcontext

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_DIR, Stage1Config


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def ensure_tokenizer_padding(tokenizer, max_seq_length: int | None = None) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if max_seq_length is not None:
        tokenizer.model_max_length = max_seq_length


def load_tokenizer(model_name_or_path: str, max_seq_length: int | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    ensure_tokenizer_padding(tokenizer, max_seq_length=max_seq_length)
    return tokenizer


def build_lora_config(config: Stage1Config) -> LoraConfig:
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules,
    )


def _load_base_model(model_name_or_path: str, config: Stage1Config, trainable: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=get_torch_dtype(config.dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = not trainable
    return model


def load_trainable_policy(
    config: Stage1Config, model_name_or_path: str = MODEL_DIR
):
    tokenizer = load_tokenizer(model_name_or_path, max_seq_length=config.max_seq_length)
    model = _load_base_model(model_name_or_path, config, trainable=True)
    model = get_peft_model(model, build_lora_config(config))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tokenizer


def load_policy_for_inference(
    checkpoint_path: str, config: Stage1Config, base_model_name: str = MODEL_DIR
):
    tokenizer_source = checkpoint_path if os.path.isdir(checkpoint_path) else base_model_name
    tokenizer = load_tokenizer(tokenizer_source, max_seq_length=config.max_seq_length)

    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        base_model = _load_base_model(base_model_name, config, trainable=False)
        model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=False)
    else:
        model = _load_base_model(checkpoint_path, config, trainable=False)

    model.config.use_cache = True
    return model, tokenizer


def adapter_disabled(model, disable: bool):
    if disable and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def count_trainable_parameters(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def save_adapter_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    config: Stage1Config,
    metadata: dict | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    payload = {
        "base_model_name": config.model_name,
        "config": config.to_dict(),
    }
    if metadata:
        payload["metadata"] = metadata

    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(payload, f, indent=2)
