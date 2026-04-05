# Runbook

This is the developer-facing runbook for the current Stage 1 setup.

The repo now uses:
- `transformers`
- `peft`
- `accelerate`
- `datasets`
- `wandb`
- `sympy`
- Modal for execution

The default model is `Qwen/Qwen3-8B`.

## 1. Preconditions

You need:
- a Modal account and CLI auth
- a Hugging Face token with access to `Qwen/Qwen3-8B`
- a Weights & Biases API key

Local Python only needs to be able to run `modal`. The heavy training dependencies are installed inside the Modal image.

If you want a local virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install modal
```

## 2. Modal Auth

Authenticate Modal:

```bash
modal token new
```

Create the required secrets:

```bash
modal secret create huggingface-secret HF_TOKEN=your_hf_token
modal secret create wandb-secret WANDB_API_KEY=your_wandb_key
```

If the secrets already exist and need updating:

```bash
modal secret delete huggingface-secret
modal secret delete wandb-secret
modal secret create huggingface-secret HF_TOKEN=your_hf_token
modal secret create wandb-secret WANDB_API_KEY=your_wandb_key
```

## 3. One-Time Setup

Populate the Modal volumes with the model and datasets:

```bash
modal run setup_data.py
```

This prepares:
- model at `/models/Qwen3-8B`
- training data at `/data/deepmath_hard_1200`
- eval data at `/data/math500`

Use this once per fresh volume, or rerun it if you intentionally want to repopulate missing artifacts.

## 4. Smoke Tests

Run these before any full training job:

```bash
modal run train.py --mode smoke_baseline --steps 10
modal run train.py --mode smoke_retro --steps 10
```

What these do:
- build the Modal image
- load the model and dataset
- run a short baseline pass
- run a short RETRO pass
- validate that rollout generation, summarisation, loss computation, and W&B logging work

Smoke tests are meant to fail fast if the stack is broken. Do not skip them on a fresh environment.

## 5. Full Training Runs

Baseline:

```bash
modal run train.py --mode baseline
```

RETRO:

```bash
modal run train.py --mode retro
```

These use the defaults from [src/config.py](/Users/an/Documents/RETRO-GRPO/src/config.py):
- `gpu = "L40S"`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 3`
- `num_generations = 4`
- `update_prompt_microbatch_size = 2`
- `num_train_epochs = 3`
- `beta = 0.0`
- `scale_rewards = "group"`
- `loss_type = "dapo"`

## 6. Evaluation

Evaluate the final baseline checkpoint:

```bash
modal run evaluate.py --checkpoint /models/baseline_final --run-name eval_baseline
```

Evaluate the final RETRO checkpoint:

```bash
modal run evaluate.py --checkpoint /models/retro_final --run-name eval_retro
```

You can also set eval batch size:

```bash
modal run evaluate.py --checkpoint /models/retro_final --run-name eval_retro --batch-size 8
```

## 7. Minimal Happy Path

If you just want the shortest correct sequence:

```bash
modal run setup_data.py
modal run train.py --mode smoke_baseline --steps 10
modal run train.py --mode smoke_retro --steps 10
modal run train.py --mode baseline
modal run train.py --mode retro
modal run evaluate.py --checkpoint /models/baseline_final --run-name eval_baseline
modal run evaluate.py --checkpoint /models/retro_final --run-name eval_retro
```

## 8. Artifacts

Expected training outputs:
- baseline checkpoints in `/models/baseline_checkpoints`
- final baseline adapter in `/models/baseline_final`
- RETRO checkpoints in `/models/retro_checkpoints`
- final RETRO adapter in `/models/retro_final`

Expected eval outputs:
- JSON results at `/models/baseline_final/eval_results.json`
- JSON results at `/models/retro_final/eval_results.json`

Each saved adapter directory also includes tokenizer files and `training_metadata.json`.

## 9. W&B Expectations

Baseline runs should log metrics like:
- reward stats
- policy loss
- ratio / clipping metrics
- baseline solve rate

RETRO runs should additionally log:
- scout solve rate
- conditioned solve rate
- scheduled conditioning rate
- failure-context rate

## 10. First Troubleshooting Knobs

If the smoke test fails, check these first.

If you hit OOM:
- first reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps` in [src/config.py](/Users/an/Documents/RETRO-GRPO/src/config.py)
- keep `update_prompt_microbatch_size = 2` so the update pass stays split into smaller backward chunks
- if `4x3` still fails, reduce `max_completion_length`

If throughput is too slow:
- if `4x3` is stable, try `6x2` while keeping `update_prompt_microbatch_size = 2`
- reduce `max_completion_length`
- reduce `rollout_summary_max_new_tokens`
- reduce `aggregate_summary_max_new_tokens`

If you want a simpler loss first:
- keep `beta = 0.0`
- keep `scale_rewards = "group"`
- keep `loss_type = "dapo"`

If you want to disable thinking for debugging:
- change `rollout_enable_thinking` in [src/config.py](/Users/an/Documents/RETRO-GRPO/src/config.py)

## 11. What Not To Change First

Avoid changing these until the smoke path is clean:
- dataset subset size
- prompt formatting
- summariser prompts
- `num_generations`
- reward extraction logic

The right first validation is stack stability, not early hyperparameter exploration.
