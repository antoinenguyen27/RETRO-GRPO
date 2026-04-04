SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Provide your final answer within \\boxed{}."
)

STAGE1_CONFIG = {
    # Model
    "model_name": "Qwen/Qwen3.5-4B",
    "max_seq_length": 2048,
    "dtype": "bfloat16",
    "load_in_4bit": False,
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Data
    "dataset_name": "zwhe99/DeepMath-103K",
    "dataset_subset_size": 1200,
    "benchmark_name": "HuggingFaceH4/MATH-500",

    # Training
    "num_generations": 4,
    "per_device_train_batch_size": 3,
    "num_train_epochs": 3,
    "max_completion_length": 512,
    "temperature": 1.0,
    "learning_rate": 5e-6,
    "beta": 0.001,
    "epsilon": 0.2,

    # RETRO-GRPO specifics
    "summariser_mode": "training_policy",
    "pipeline_mode": "in_step",
    "annealing": False,

    # Infrastructure
    "gpu": "L4",
    "wandb_project": "retro-grpo-poc",
}

# Volume paths (inside Modal containers)
MODEL_VOLUME_PATH = "/models"
DATA_VOLUME_PATH = "/data"
MODEL_DIR = f"{MODEL_VOLUME_PATH}/Qwen3.5-4B"
TRAIN_DATA_DIR = f"{DATA_VOLUME_PATH}/deepmath_hard_1200"
EVAL_DATA_DIR = f"{DATA_VOLUME_PATH}/math500"
