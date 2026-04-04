# Training Memory And Batching

This note explains the custom memory-path changes in the stage-1 trainer and the batch layouts we use for GRPO.

It is written against the current runtime in [src/trainer.py](/Users/an/Documents/RETRO-GRPO/src/trainer.py), [src/grpo.py](/Users/an/Documents/RETRO-GRPO/src/grpo.py), and [src/config.py](/Users/an/Documents/RETRO-GRPO/src/config.py).

## Why This Exists

The stock `Accelerate + bf16` path is not a good fit for this project on a single GPU because:

- Accelerate wraps `forward()` with an fp32 output conversion after autocast.
- The trainer computes `old_logprobs` before the update step and used to do that with gradients enabled.
- The GRPO logprob path used `F.log_softmax(logits, dim=-1)`, which materialized a second full `[B, T, V]` tensor.

Those choices are generic and safe, but expensive for a Qwen3-8B GRPO trainer with long sequences and multiple rollouts per prompt.

## What We Changed

### 1. Removed output widening while keeping autocast

After `Accelerator.prepare(...)`, the trainer now strips only the outer `ConvertOutputsToFp32` layer from the unwrapped model in [src/trainer.py](/Users/an/Documents/RETRO-GRPO/src/trainer.py).

What this means:

- We still use Accelerate autocast for bf16 compute.
- We no longer widen the full model output back to fp32.
- We still explicitly upcast the small tensors that matter for numerically sensitive policy math.

This is intentionally narrower than `unwrap_model(..., keep_fp32_wrapper=False)`, which would also remove the autocast wrapper.

### 2. `old_logprobs` and `ref_logprobs` now run under `torch.no_grad()`

In [src/trainer.py](/Users/an/Documents/RETRO-GRPO/src/trainer.py), the old-policy pass used to be computed in eval mode but with autograd still enabled.

That is wasteful because:

- those logprobs are detached immediately after computation
- they are not part of the backward pass

The trainer now computes:

- `old_logprobs` under `torch.no_grad()`
- `ref_logprobs` under `torch.no_grad()` when `beta > 0`

This is the conservative compatibility choice. We use `no_grad()`, not `inference_mode()`.

### 3. Repacked logprob batches to left padding and restricted logits to the completion suffix

The packed rollout tensors in [src/grpo.py](/Users/an/Documents/RETRO-GRPO/src/grpo.py) are now left-padded instead of right-padded.

Why:

- the real training signal only needs completion-token logprobs
- with left padding, every sequence ends at the same right edge
- that means the completion suffix is aligned across the whole batch

The logprob forward now passes `logits_to_keep` to Qwen3 so the model only emits the final suffix window needed for completion-token scoring, instead of full-sequence logits for every padded and prompt position.

The kept suffix length is:

```text
max_completion_length_in_batch + 1
```

The extra `+1` is needed because the first completion token depends on the logit at the final prompt position.

### 4. Replaced full `log_softmax` materialization with gathered logits plus chunked `logsumexp`

Even after restricting logits to the suffix window, the old code would still have materialized a second full `[B, T, V]` tensor via:

```python
F.log_softmax(logits, dim=-1).gather(...)
```

That creates a second full `[B, T, V]` tensor.

The current code instead uses the identity:

```text
log_softmax(logits)[target] = logits[target] - logsumexp(logits)
```

Implementation details:

- gather the selected token logits first
- compute `logsumexp` over the vocab axis
- subtract to get the selected-token logprob

The `logsumexp` reduction is done in fp32, but in vocab chunks, so we do not allocate a full fp32 copy of the logits tensor at once.

### 5. Kept fp32 only where it still helps

We still use fp32 for:

- reward / advantage statistics
- gathered token logprobs
- policy ratio math (`log_ratio`, `ratio`, clipping, KL)
- Qwen internal stability casts such as RMSNorm and attention softmax

We do not use fp32 for the full `[B, T, V]` output tensor anymore.

## Batch Vocabulary

This trainer has four different counting units.

### Prompt

One training example from the dataset. A prompt is the unique problem/question.

### Rollout or generation

One sampled completion for one prompt. We keep `num_generations = 4`, so each prompt produces 4 sampled completions.

### Microbatch

One dataloader batch. Its size is `per_device_train_batch_size`.

If `per_device_train_batch_size = 6`, then one microbatch contains 6 unique prompts.

### Accumulation steps

How many microbatches we process before one optimizer update.

If `gradient_accumulation_steps = 2`, then we process 2 microbatches and then call `optimizer.step()`.

### Optimizer step

One full GRPO update cycle:

- collect `gradient_accumulation_steps` microbatches
- generate rollouts for each microbatch
- compute `old_logprobs`
- compute rewards and advantages
- run the update pass
- call `optimizer.step()`

## What `6x2` Means

`6x2` means:

- `per_device_train_batch_size = 6`
- `gradient_accumulation_steps = 2`
- `num_generations = 4`

So:

- each microbatch contains 6 unique prompts
- each prompt produces 4 rollouts
- each microbatch therefore contains 24 rollout sequences
- one optimizer step sees 12 unique prompts total
- one optimizer step sees 48 rollout sequences total

Important:

- `6x2` does not mean 6 rollouts and 2 prompts
- the rollout count is controlled separately by `num_generations`

## Current Default

The current default runtime in [src/config.py](/Users/an/Documents/RETRO-GRPO/src/config.py) is:

- `gpu = "L40S"`
- `per_device_train_batch_size = 6`
- `gradient_accumulation_steps = 2`
- `num_generations = 4`

This keeps the effective prompt batch at 12 while reducing accumulation overhead relative to the older `3x4` plan.

## Supported Layouts

| Layout | Prompts / microbatch | Accumulation steps | Effective prompt batch | Rollouts / microbatch | Rollouts / optimizer step | Optimizer steps for 1,200 problems x 3 epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3x4` | 3 | 4 | 12 | 12 | 48 | 300 |
| `4x3` | 4 | 3 | 12 | 16 | 48 | 300 |
| `6x2` | 6 | 2 | 12 | 24 | 48 | 300 |
| `4x4` | 4 | 4 | 16 | 16 | 64 | 225 |
| `5x4` | 5 | 4 | 20 | 20 | 80 | 180 |

How to read this table:

- If the effective prompt batch stays at 12, the training budget stays the same and only the packing changes.
- If the effective prompt batch increases to 16 or 20, the default epoch-based training loop automatically reduces optimizer steps.

## Which Layout To Use

### `3x4`

Use this only as the old conservative baseline. It has the most accumulation overhead.

### `4x3`

Use this when you want the safest apples-to-apples improvement over `3x4`.

### `6x2`

Use this when you want the same effective prompt batch as `3x4`, but with better throughput on a GPU that can fit the larger physical microbatch.

This is the best default if memory headroom is available.

### `4x4`

Use this when you intentionally want a larger effective prompt batch and fewer optimizer steps.

### `5x4`

Use this only as a larger-batch experiment. It changes the optimization regime more materially.

## How To Change Batch Layout

Edit these fields in [src/config.py](/Users/an/Documents/RETRO-GRPO/src/config.py):

- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- keep `num_generations = 4` unless you intentionally want to change GRPO grouping

The trainer computes total optimizer steps from:

```text
ceil(dataset_size / per_device_train_batch_size)
-> ceil(micro_steps_per_epoch / gradient_accumulation_steps)
-> multiply by num_train_epochs
```

That logic lives in [src/trainer.py](/Users/an/Documents/RETRO-GRPO/src/trainer.py).

## Memory Intuition

Accumulation is sequential in this trainer.

That means a `6x2` run does not hold 48 rollout sequences live at once. The expensive forward/backward peak is dominated by one physical microbatch, which is 24 sequences for `6x2`.

So:

- `3x4` peaks like 12 live sequences
- `4x3` peaks like 16 live sequences
- `6x2` peaks like 24 live sequences

The main benefit of moving from `3x4` to `6x2` is:

- fewer microbatch loops per optimizer step
- better GPU utilization
- less accumulation overhead

The main cost is:

- higher peak VRAM per microbatch

## Practical Recommendation

For the current single-GPU Modal runtime:

- keep `num_generations = 4`
- use `6x2` when the GPU fits it comfortably
- fall back to `4x3` if `6x2` is too tight

If you change the batch layout, remember to compare runs using the right unit:

- prompt groups per optimizer step
- total prompt budget across training

Not just raw optimizer-step count.
