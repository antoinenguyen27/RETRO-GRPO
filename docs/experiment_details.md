# Experiment Details: RETRO-GRPO Stage Plan

## Budget Constraint: $30 Modal Credits

---

## 1. Purpose of This Document

This document is the operational plan for the proof-of-concept experiments. It is intentionally narrower than the research spec.

The experiment order is staged:

1. **Stage 1:** prove that descriptive failure-conditioning improves training at all.
2. **Stage 2:** add annealing with **batch consistency**.
3. **Stage 3:** add the **per-prompt coin-flip** annealing variant.

We do **not** treat annealing as part of the first proof experiment. Stage 1 answers the first-order question: does the method produce uplift on hard math RL training?

---

## 2. Stage 1: The First Real Question

**Question:** Does prepending descriptive summaries of failed attempts improve GRPO training on hard math problems?

**Comparison:**
- **Baseline:** standard GRPO
- **Method:** RETRO-GRPO with descriptive failure summaries

**No annealing in stage 1.** The point is to validate the mechanism before optimising the withdrawal schedule.

### Stage-1 rollout logic

For each prompt in the batch:

1. Generate 4 unconditioned scout rollouts.
2. Score them with deterministic answer verification.
3. Summarise the failed scout attempts from that scout set, then aggregate
   those descriptions into one summary.
4. Prepend the descriptive summary to the prompt.
5. Generate 4 conditioned rollouts.
6. Train on the conditioned rollouts only.

Stage 1 does **not** use scout-success gating. The stage-1 RETRO-GRPO path always proceeds through summarisation and conditioned rollout generation; the point is to test the conditioned mechanism directly before adding annealing or routing shortcuts.

---

## 3. Domain and Verification

### Domain

**Math only** for the proof of concept.

Why:
- exact answer verification,
- shorter rollouts than agentic tasks,
- structured failure modes that are easy to summarise,
- clean comparison against existing reasoning-RL work.

### Verification

Stage 1 uses **deterministic verification**, not LLM judges.

- Extract the final answer.
- Normalize.
- Compare against the ground truth answer.
- Add lightweight symbolic-equivalence handling only if necessary for formatting edge cases.

This keeps reward computation cheap, objective, and easy to debug.

---

## 4. Model and Training Stack

### Model

**Qwen3-8B**

This is the default operational model for the PoC. We do not discuss fallback models in this document.

### Fine-tuning method

**bf16 LoRA via a custom stack**

- `transformers + peft + accelerate`
- No TRL in the runtime training path.
- No Unsloth.
- No QLoRA in the main plan.
- LoRA rank: **32**
- Summariser mode: **training policy**
- Framing variation: retained

### Compute

**Modal L4** by default.

This is the default GPU for the PoC. If smoke tests show that Qwen3-8B plus the two-phase rollout path does not fit comfortably on an L4, the fallback is to move to **L40S** only after that failure is observed.

Reasoning:
- single-GPU operation keeps the custom trainer simple,
- the L4 is the default operating point for stage 1,
- we only pay for a higher-tier GPU if the smoke test proves it necessary.

### Framework

**Custom in-repo trainer built on `transformers`, `peft`, `accelerate`, `datasets`, `wandb`, and `sympy`.**

The rollout choreography, reward verification, and GRPO loss are implemented locally. This keeps the method stable under Qwen3 and makes the RETRO-specific generation path fully under our control.

---

## 5. Training Data

### Training dataset

**DeepMath-103K**, hard-filtered for stage 1.

### Exact stage-1 training subset

Use **1,200 problems**.

### Filtering target

Keep problems where the base policy is still clearly struggling, operationally something like:

- **pass@4 ≤ 0.25**

The exact threshold can be finalized after the filtering pass, but the intent is fixed: keep the training set in the hard-problem regime so the conditioning mechanism is exercised frequently.

### Why hard-filter instead of using the natural distribution

Stage 1 is not trying to prove broad transfer or annealing behaviour. It is trying to answer a narrower question:

**does failure conditioning help on hard problems?**

A hard-filtered subset is more informative than broad natural difficulty because it:
- keeps conditioning active on a large share of prompts,
- reduces dilution from already-easy problems,
- makes the learning-curve comparison between baseline and RETRO-GRPO sharper.

### Breadth vs depth

For stage 1, favor **breadth with low epoch count** over narrow/deep training.

Reasoning:
- base GRPO generally benefits more from breadth when compute is limited,
- broader coverage gives a wider strategy surface,
- it provides a cleaner proof test of whether RETRO-GRPO helps in general on hard math rather than on a narrow repeated slice.

If stage 1 shows real uplift, narrow/deep training can be tested later as a follow-up regime.

---

## 6. Benchmark Choice

### Single stage-1 benchmark

Use **MATH-500 only**.

### Why MATH-500

It is the best single benchmark under budget because it balances:
- hard enough problems to make conditioning relevant,
- enough examples to produce a stable readout,
- easier analysis than tiny extreme-hard sets such as AIME alone.

### Reporting strategy

Even with one benchmark, report two views:

1. **Overall MATH-500**
2. **Hard slice of MATH-500** (headline view)

The hard slice is where we most expect RETRO-GRPO to help.

### Not in stage 1

- GSM8K is removed from the core stage-1 evidence package.
- AIME is not the sole stage-1 benchmark because it is too small and noisy for the first proof experiment.

---

## 7. Batch Size and Step Count

Batch size must be treated explicitly in GRPO.

### Default stage-1 batching

- `per_device_train_batch_size = 3`
- `gradient_accumulation_steps = 4`
- `num_generations = 4`
- **effective prompt batch size = 12**
- **effective conditioned completions per optimizer step = 48**
- target unique prompts per generation cycle: **3**

This avoids the too-small effective-batch regime and is cleaner than the earlier implicit batch-8 plan.
The key accounting unit is prompt groups, not repeated prompt slots. Each microbatch contains `3` prompt groups and `4` rollouts per prompt. Gradient accumulation over `4` microbatches yields `12` prompt groups per optimizer step.

### Stage-1 optimizer-step count

With:
- 1,200 training problems,
- 3 epochs,
- effective prompt batch size 12,

we get:

- **300 optimizer steps**

This is the number to use in the operational plan.

---

## 8. Core Hyperparameters for Stage 1

```python
stage1_config = {
    "model": "Qwen3-8B",
    "gpu": "L4",
    "stack": "transformers_peft_accelerate",
    "finetuning": "bf16_lora",
    "lora_rank": 32,
    "dataset": "DeepMath-103K_hard_filtered_1200",
    "epochs": 3,
    "benchmark": "MATH-500",
    "per_device_train_batch_size": 3,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,
    "effective_prompt_batch_size": 12,
    "effective_conditioned_completions": 48,
    "optimizer_steps": 300,
    "max_completion_length": 1024,
    "rollout_enable_thinking": True,
    "rollout_temperature": 0.6,
    "rollout_top_p": 0.95,
    "rollout_top_k": 20,
    "summary_enable_thinking": False,
    "learning_rate": 5e-6,
    "beta": 0.0,
    "epsilon": 0.2,
    "scale_rewards": "group",
    "loss_type": "dapo",
    "summariser_mode": "training_policy",
    "pipeline_mode": "in_step",
    "annealing": False,
}
```

Notes:
- `max_completion_length = 1024` is the stage-1 rollout budget.
- `num_generations = 4` is retained as the main budget lever.
- The in-step two-phase pipeline with hierarchical summarisation remains the default.
- Qwen3 rollout generation uses thinking mode with the model-card sampling defaults.
- Summary generation disables thinking and runs greedily.
- Prompt inputs are left-truncated as needed so prompt plus completion stays within the model context window.
- `beta = 0.0` removes reference-model overhead in stage 1.

### Reward scaling and loss reduction knobs

The custom trainer exposes the GRPO reduction choices directly.

- `scale_rewards = "group"`: subtract each prompt-group mean and divide by that prompt-group standard deviation. This is the default GRPO normalisation.
- `scale_rewards = "batch"`: subtract each prompt-group mean but divide by the standard deviation over the whole optimizer-step batch.
- `scale_rewards = "none"`: subtract each prompt-group mean and do not divide by a standard deviation term.

- `loss_type = "grpo"`: average token objectives within each completion, then average completions. This is the original GRPO reduction and is length-biased.
- `loss_type = "dapo"`: sum token objectives and normalise by the total number of active completion tokens. This is the stage-1 default.
- `loss_type = "dr_grpo"`: sum token objectives and normalise by `num_sequences * max_completion_length`, which removes sequence-length dependence from the denominator.

Stage 1 defaults to:
- `scale_rewards = "group"`
- `loss_type = "dapo"`

---

## 9. Stage 1 Implementation Notes

### Batching behaviour

The rollout loop should be implemented in batched form, not prompt-by-prompt.

For a prompt batch of `B` prompts:

1. batched scout generation,
2. batched scout scoring,
3. build first-pass summariser prompts for the failed scouts from each prompt's scout set,
4. run first-pass summarisation,
5. build aggregate summariser prompts for every prompt,
6. run aggregate summarisation,
7. batched conditioned generation for every prompt in the batch,
8. assemble the final training batch,
9. standard GRPO loss and update.

### Memory handling

Scouts do **not** need to be retained once they have been scored and, where relevant, converted into rollout summaries. This keeps the memory profile close to standard GRPO, aside from the extra sequential summarisation calls.

### Summaries

Summaries stay:
- descriptive,
- detailed enough to preserve approach structure,
- non-prescriptive,
- non-evaluative.

We retain framing randomization to avoid brittle dependence on one wrapper format.

---

## 10. Stage 2 and Stage 3

These stages are explicitly deferred until stage 1 shows uplift.

### Stage 2: batch-consistent annealing

Add annealing with a **global EMA-driven conditioning probability** and a **single decision for the full batch**.

Interpretation:
- either the whole batch runs the conditioning path,
- or the whole batch runs standard GRPO.

This gives smoother withdrawal while preserving batch consistency.

### Stage 3: per-prompt coin-flip annealing

Add the per-prompt coin-flip variant on top of the global schedule.

Interpretation:
- the global EMA controls the overall conditioning rate,
- prompts in the same batch may diverge in whether they receive conditioning.

This stage tests whether mixed conditioned/unconditioned prompts help enough to justify the extra complexity and throughput cost.

---

## 11. Budget and Throughput

### Important correction

Modal cost should be treated as:
- GPU cost,
- CPU cost,
- memory cost,

not GPU-only.

So the earlier rough budget math was too optimistic.

### Operational consequence

The stage-1 plan must fit:
- one filtering pass,
- one smoke test,
- one baseline GRPO run,
- one RETRO-GRPO run,
- evaluation,

inside a much tighter real budget envelope than the earlier T4-based draft implied.

### Practical stance

Because true throughput depends on:
- prompt length,
- completion length,
- fraction of prompts taking the conditioned branch,
- fast-inference behaviour,
- and trainer overhead,

we should treat wall-clock estimates as **smoke-test validated**, not fixed on paper.

The exact-stage-1 dataset size of **1,200** is chosen partly to create budget headroom for that uncertainty.

---

## 11A. Telemetry

Telemetry for all stages uses **Weights & Biases (W&B) only**. It is the single experiment-tracking backend and system of record for configs, metrics, and run comparisons.

The setup should remain simple and consistent across runs:
- one W&B project for the PoC,
- separate run names / tags for `baseline_grpo` and `retro_grpo`,
- stage labels recorded in config,
- no separate tracking stack in parallel.

### What to log

Log the usual trainer metrics plus the method-specific signals that matter for comparison:
- run config and experiment tag,
- code version / commit hash if available,
- reward statistics,
- loss / KL / clipping metrics,
- step time and throughput,
- GPU memory and utilization if available,
- evaluation scores on the fixed MATH-500 subset.

For RETRO-GRPO specifically, also log:
- scout solve rate,
- conditioned rollout solve rate,
- scheduled conditioning rate,
- failure-context rate.

For stage 1, the scheduled conditioning rate for RETRO-GRPO is simply **1.0** because there is no annealing and no scout-success routing. The failure-context rate may still be below `1.0` on prompts where all scouts succeed and no failure summary is prepended. For baseline GRPO, conditioning metrics can be omitted or logged as `0.0` / `N/A` consistently.

---

## 12. What Is Removed from the Old Plan

These are no longer stage-1 defaults:

- T4 as the main GPU
- Qwen2.5-3B
- 4-bit QLoRA
- the mixed easy/medium/hard training split
- prescriptive framing as an early core ablation
- annealing in the first proof experiment
- GSM8K in the main stage-1 benchmark package
- loose training-size ranges such as 1000-1500
- the implicit effective-batch-size-8 assumption

---

## 13. Stage-1 Anchor Summary

**Stage 1 uses Qwen3-8B with bf16 LoRA on a custom `transformers + peft + accelerate` stack on an L4, trains baseline GRPO vs RETRO-GRPO on a hard-filtered 1,200-problem DeepMath subset for 3 epochs at effective prompt batch size 12 (300 optimizer steps), evaluates on MATH-500 only, and uses no annealing.**
