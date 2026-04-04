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
3. If **any** scout succeeds, use the scout rollouts directly as the training rollouts.
4. If **all** scouts fail, summarise the failed attempts.
5. Prepend the descriptive summary to the prompt.
6. Generate 4 conditioned rollouts.
7. Train on the conditioned rollouts only.

The "any scout succeeds" branch is an **efficiency shortcut**, not annealing.

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

**Qwen3.5-4B**

This is the default operational model for the PoC. We do not discuss fallback models in this document.

### Fine-tuning method

**bf16 LoRA via Unsloth**

- No QLoRA in the main plan.
- LoRA rank: **32**
- Summariser mode: **training policy**
- Framing variation: retained

### Compute

**Modal L4**

This is the default GPU for the PoC.

Reasoning:
- enough VRAM headroom for Qwen3.5-4B bf16 LoRA,
- lower operational risk than squeezing the run onto a T4,
- matches the revised stack choice.

### Framework

**Unsloth + TRL GRPOTrainer** with a light custom modification to the rollout-generation path.

The core GRPO update remains standard. The custom code only changes how training rollouts are generated.

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

- `num_generations = 4`
- **effective batch size = 12**
- target unique prompts per generation cycle: **3**

This avoids the too-small effective-batch regime and is cleaner than the earlier implicit batch-8 plan.

### Stage-1 optimizer-step count

With:
- 1,200 training problems,
- 3 epochs,
- effective batch size 12,

we get:

- **300 optimizer steps**

This is the number to use in the operational plan.

---

## 8. Core Hyperparameters for Stage 1

```python
stage1_config = {
    "model": "Qwen3.5-4B",
    "gpu": "L4",
    "finetuning": "bf16_lora",
    "lora_rank": 32,
    "dataset": "DeepMath-103K_hard_filtered_1200",
    "epochs": 3,
    "benchmark": "MATH-500",
    "num_generations": 4,
    "effective_batch_size": 12,
    "optimizer_steps": 300,
    "max_completion_length": 512,
    "temperature": 1.0,
    "learning_rate": 5e-6,
    "beta": 0.001,
    "epsilon": 0.2,
    "summariser_mode": "training_policy",
    "pipeline_mode": "in_step",
    "annealing": False,
}
```

Notes:
- `max_completion_length = 512` remains the default budget-conscious choice.
- `num_generations = 4` is retained as the main budget lever.
- The in-step two-phase pipeline remains the default.

---

## 9. Stage 1 Implementation Notes

### Batching behaviour

The rollout loop should be implemented in batched form, not prompt-by-prompt.

For a generation batch of `B` prompts:

1. batched scout generation,
2. batched scout scoring,
3. build summariser prompts only for all-failed prompts,
4. run summarisation,
5. batched conditioned generation for those prompts,
6. assemble the final training batch,
7. standard GRPO loss and update.

### Memory handling

Scouts do **not** need to be retained once they have been scored and, where relevant, converted into summary input. This keeps the memory profile close to standard GRPO, aside from the extra sequential generation calls.

### Summaries

Summaries stay:
- descriptive,
- short,
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

**Stage 1 uses Qwen3.5-4B with bf16 LoRA via Unsloth on an L4, trains baseline GRPO vs RETRO-GRPO on a hard-filtered 1,200-problem DeepMath subset for 3 epochs at effective batch size 12 (300 optimizer steps), evaluates on MATH-500 only, and uses no annealing.**
