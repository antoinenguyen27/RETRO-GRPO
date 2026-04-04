from dataclasses import asdict, dataclass, field


SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Provide your final answer within \\boxed{}."
)


@dataclass(slots=True)
class Stage1Config:
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    max_seq_length: int = 2048
    dtype: str = "bfloat16"
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Data
    dataset_name: str = "zwhe99/DeepMath-103K"
    dataset_subset_size: int = 1200
    benchmark_name: str = "HuggingFaceH4/MATH-500"

    # Optimisation
    num_generations: int = 4
    per_device_train_batch_size: int = 3
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float | None = None
    num_iterations: int = 1
    scale_rewards: str = "group"
    loss_type: str = "dapo"
    mask_truncated_completions: bool = False

    # Rollout budgets
    max_completion_length: int = 1024
    rollout_summary_max_new_tokens: int = 384
    aggregate_summary_max_new_tokens: int = 512

    # Qwen3 rollout generation
    rollout_enable_thinking: bool = True
    rollout_temperature: float = 0.6
    rollout_top_p: float = 0.95
    rollout_top_k: int = 20
    rollout_min_p: float = 0.0

    # Summary generation
    summary_enable_thinking: bool = False
    summary_do_sample: bool = False

    # Evaluation generation
    eval_enable_thinking: bool = True
    eval_do_sample: bool = True
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_top_k: int = 20
    eval_min_p: float = 0.0

    # RETRO specifics
    summariser_mode: str = "training_policy"
    pipeline_mode: str = "in_step"
    annealing: bool = False

    # Logging and infra
    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 3
    dataloader_num_workers: int = 0
    gpu: str = "L4"
    wandb_project: str = "retro-grpo-poc"

    @property
    def effective_prompt_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        return asdict(self)


STAGE1_CONFIG = Stage1Config()


# Volume paths (inside Modal containers)
MODEL_VOLUME_PATH = "/models"
DATA_VOLUME_PATH = "/data"
MODEL_DIR = f"{MODEL_VOLUME_PATH}/Qwen3-8B"
TRAIN_DATA_DIR = f"{DATA_VOLUME_PATH}/deepmath_hard_1200"
EVAL_DATA_DIR = f"{DATA_VOLUME_PATH}/math500"
