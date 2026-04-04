# Failure-Conditioned Exploration for Reinforcement Learning in LLMs

## Research Specification Document — v3

---

## 1. Core Idea

We propose a method that improves exploration in RL-based LLM training by conditioning rollout generation on descriptive summaries of prior failed attempts. Rather than telling the model what to do (positive conditioning, as in iGRPO) or what not to do (prescriptive negative conditioning), we provide a factual narrative of what was tried and how it ended. The model's own reasoning determines what to suppress and what to explore instead.

The failure information enters the training loop through the **context window**, not through the **gradient**. This architectural separation means exploration guidance (what to try next) and exploitation learning (which strategies to reinforce) operate through independent channels. The context steers exploration without modifying parameters; GRPO reinforces discoveries through standard policy gradient. The conditioning is annealed over training, and the model internalises failure-avoidance into its weights through on-policy learning.

---

## 2. Why We Believe This Works

### 2.1 Theoretical Foundation

**In-context learning transfers to weights.** ICRL (Ye et al., 2026) demonstrates that in-context demonstrations during RL rollouts can be gradually removed while the model retains the learned behaviour in its parameters. SDFT (Shenfeld et al., 2026) formalises this: a model conditioned on context acts as a more capable teacher, and on-policy distillation transfers that capability to the unconditioned student. Our method uses the same mechanism — the conditioned distribution π_θ(y|x, c) is more capable (it avoids known-bad strategies), and GRPO reinforcement of successful conditioned rollouts transfers that capability to π_θ(y|x).

**Negative signal preserves exploration.** NSR (Zhu et al., 2025) proves that training with only negative samples preserves the full pass@k spectrum, while positive-only training (PSR) degrades pass@k at large k through over-sharpening. Our method provides negative signal (information about failures) through context rather than gradient, which should preserve exploration at least as well as NSR while avoiding NSR's collateral damage to useful sub-strategies that happen to co-occur with failures.

**The conditioned distribution is strictly more capable than the unconditioned prior.** NSR's ceiling is the unconditioned prior — it can only redistribute mass that already exists. Our method operates on the conditioned distribution, which can express strategies the model would never produce unconditioned. A model that always tries symbolic integration unconditioned might, when told "previous attempts used symbolic integration and did not succeed," try numerical methods it would never have selected otherwise. This means some problems that are Type 2 for NSR (zero pass@k unconditioned) become Type 1 under our conditioning (non-zero pass@k conditioned).

### 2.2 Mechanism: Semantic Exploration Broadening

The conditioning performs informed, semantic-level exploration broadening. Define the effective exploration set E_t(x) as the set of qualitatively distinct strategies the model samples from on prompt x at step t.

Under standard GRPO, |E_t(x)| shrinks over training as the policy concentrates on strategies that have received positive advantage. This is the standard entropy collapse trajectory.

Under our conditioning, the model reads descriptive summaries of previously tried strategies. Its pretrained understanding of narrative context — "these things were tried and didn't work" — leads it to redistribute probability mass away from the described strategy classes and toward alternatives. The conditioned exploration set E_t(x, c) is enriched for strategies NOT resembling those in c, including low-probability strategies from the tail of the unconditioned distribution.

This enrichment is **targeted** rather than uniform. An entropy bonus increases probability on all low-probability strategies indiscriminately. Our conditioning specifically suppresses strategies that are known to fail, concentrating the redistributed mass on strategies that have not been tried — which are, in a Bayesian sense, enriched for being correct.

Critically, the suppression is **semantic** — operating at the level of strategy classes, not token sequences. The model reads "set up polar coordinates, applied substitution on the inner integral" and suppresses the entire class of polar-coordinate-with-substitution approaches, not just the specific token sequence of one failed rollout.

### 2.3 Soft Framing: Why Descriptive Beats Prescriptive

We use soft, descriptive framing ("here's what was tried, it didn't succeed") rather than hard, prescriptive framing ("don't use polar coordinates"). This is superior for three reasons.

**Adaptive suppression granularity.** Hard framing suppresses at whatever granularity the judge chose. "Don't use polar coordinates" kills the entire strategy class, including productive variants. Soft framing lets the model calibrate its own suppression — it might infer "polar was fine, the substitution was the issue" or "the whole polar approach is wrong here," depending on its own understanding of the problem. The suppression adapts to the model's knowledge rather than being fixed by the judge's categorisation.

**Richer gradient signal.** Conditioned rollouts under soft framing contain the model's own meta-reasoning: "the previous attempt used substitution on the inner integral and didn't work, so let me consider whether integration order matters." When GRPO reinforces successful conditioned rollouts, it reinforces this meta-reasoning, not just strategy execution under an external constraint. This meta-reasoning transfers to the unconditioned setting — the model learns to reflect on its own approach choices, a skill that doesn't require context to trigger.

**Better transfer to unconditioned inference.** Hard framing trains instruction-following ("when told to avoid X, try alternatives"). Soft framing trains reasoning ("when I see evidence that an approach class failed, reconsider my strategy"). The latter skill activates within the model's own chain of thought at inference time, without any external context.

---

## 3. What This Enables Over Existing Methods

### 3.1 vs. Standard GRPO

GRPO on hard problems (low solve rate) produces rollouts that are overwhelmingly failures. Most rollout compute is wasted — failures with similar failure modes generating near-zero variance in advantages. Our method redirects this wasted exploration toward unexplored strategy regions, increasing the probability that at least some rollouts discover correct approaches. The rollout budget is unchanged; the fraction of informative rollouts increases.

On problems where GRPO already achieves non-zero but low solve rates, the same dynamic applies less dramatically. The model may be solving the problem one way but wasting most rollouts on a dominant failure mode. The conditioning suppresses that mode, potentially uncovering additional solution strategies, improving both pass@1 (more likely to find a solution) and pass@k (more diverse solutions).

### 3.2 vs. iGRPO

iGRPO conditions on the **best draft** (highest-reward rollout) in an in-step two-phase structure. We adopt the same in-step two-phase pipeline — this is a deliberate design choice validated by iGRPO's results, not a structural contribution. The contribution is in the information-theoretic difference: what the conditioning contains and how it affects exploration.

iGRPO's best-draft conditioning is an attractor — it anchors stage 2 exploration to the neighbourhood of the best attempt. When the best attempt is correct (medium-difficulty problems), this works well: the model refines a known-good strategy. When the best attempt is still a failure (hard problems), the attractor is counterproductive: the model refines around the least-bad failure rather than seeking qualitatively different approaches.

Our conditioning is a soft repeller. It pushes exploration away from known-bad regions without specifying where to go. On hard problems, this is strictly more appropriate — the model needs to discover new strategies, not refine existing failures.

The two methods are complementary across the difficulty spectrum:
- Hard problems (solve rate near zero): failure conditioning only. iGRPO's best-draft is a failed trajectory; conditioning on it anchors to the wrong region.
- Medium problems (solve rate 0.1-0.5): iGRPO's best-draft provides direction; failure conditioning provides repulsion from common failure modes. Both signals are compatible (the success is by definition not in the failure mode space).
- Easy problems (solve rate >0.5): standard GRPO suffices. Neither method adds value.

Our method specifically fills the gap where iGRPO fails — the hard-problem regime.

### 3.3 vs. Critique-GRPO

Critique-GRPO and our method share a high-level shape — an initial set of rollouts, a natural-language signal derived from them, and a second generation phase — but they are built around different failure theories and therefore different learning dynamics.

**Failure theory.** Critique-GRPO is built for the regime where the model often has the latent capability to solve the problem once it is given a diagnosis of what went wrong in a specific failed attempt. Our method is built for the regime where rollout budget is repeatedly spent on the same dominant failure modes, so the bottleneck is not diagnosis of one attempt but redirection of exploration away from already-tried regions.

**Where the information enters.** Critique-GRPO injects information after an initial attempt has already been generated: the critique is used to produce a refinement. Our method injects information before generation begins: the failure summary is prepended to the prompt and shapes strategy selection from token 1. This means Critique-GRPO primarily modifies correction behaviour, while our method primarily modifies search behaviour.

**Scope of information.** Critique-GRPO is 1:1. Each response receives its own critique, and each refinement is conditioned only on its own parent attempt and that critique. There is no cross-rollout aggregation. Our method aggregates across failed scouts into a single summary, so every conditioned rollout sees population-level evidence about what the current policy family has already tried on that prompt. This is the core information-structure difference.

**What stage 2 is doing.** Critique-GRPO's second phase is anchored refinement: the model is asked to repair a particular failed trajectory in light of a diagnosis. Our second phase is fresh first-pass generation under contextualised failure evidence: the model is not repairing one attempt, but generating a new solution from scratch after reading what strategy classes were already explored.

**What receives gradient.** Critique-GRPO applies gradient to both the initial responses and the refinement samples, using modified weighting for the refinement component. This gives it an implicit anti-dependence mechanism: the model continues to practise unconditioned generation every step, while refinement-discovered behaviours can leak into the base policy through shared parameters. Our method, by default, applies standard GRPO only to the conditioned rollouts when conditioning triggers. This cleanly concentrates gradient on exploration-improved samples, but it also means the return to unconditioned competence must be handled explicitly by annealing.

**Architectural and statistical stance.** Critique-GRPO mixes samples from two generative processes in one update group and compensates with shaping: unconditioned initial responses plus critique-conditioned refinements, with special treatment for refinement gradients. Our method leaves GRPO itself untouched and instead changes the rollout distribution: standard on-policy GRPO is applied to whichever distribution is active for that prompt in that step, unconditioned or failure-conditioned. In that sense, Critique-GRPO modifies the optimisation procedure, while we modify the information presented to the policy before sampling.

**Why these choices make sense for each method.** Critique-GRPO's design is coherent if the main problem is correction of a specific failed reasoning trace: per-response critique is maximally informative, anchored refinement is the shortest path to a corrected answer, and joint gradient prevents the model from drifting into dependence on a training-only signal. Our design is coherent if the main problem is repeated exploration of the same failing strategy class: cross-rollout aggregation is necessary to expose that repetition, pre-generation context is the natural place to redirect search, and scout exclusion spends gradient budget on the samples most likely to contain new information.

**Honest assessment.** We do not claim that Critique-GRPO cannot produce strategy-level departures. Diversity across independent refinements can absolutely yield qualitatively new strategies, and successful departures can be reinforced. Our claim is narrower and more defensible: cross-rollout descriptive conditioning should direct a larger fraction of second-phase rollout budget away from already-tried failure modes on any given step, which should matter most in the hard-problem regime where solve rates are near zero and failures are highly concentrated.

**Practical distinction.** Critique-GRPO depends on a judge or reward model capable of producing useful critiques of specific responses. Our summaries require only factual trajectory description across failed attempts. This is a simpler supervision problem and keeps the method focused on exploration guidance rather than error diagnosis.

### 3.4 vs. NSR

NSR suppresses failed trajectories through negative gradient signal. This is effective but operates at the token-sequence level — every token in a failed trajectory gets its probability reduced, including tokens from useful sub-strategies that happened to co-occur with the failure. NSR's "collateral damage" erodes useful reasoning patterns over many updates.

Our method provides failure information through context, not gradient. The weights are updated only by GRPO on whatever the model discovers under conditioning. No trajectory ever receives negative gradient pressure from the failure information — only from standard GRPO advantage computation. Useful sub-strategies within failed trajectories are preserved because the failure conditioning is suppressing at the semantic strategy level, not the token level.

Additionally, NSR's redistribution follows the parameter geometry (mass flows to strategies that are "nearby" in parameter space to the suppressed ones). Our redistribution follows the model's semantic understanding (mass flows to strategies the model considers "different approaches"). The semantic redistribution is more likely to reach qualitatively different strategy classes.

### 3.5 The Hard-Problem Regime — Honest Assessment

We should not overstate the zero-reward argument. Even at zero reward across all rollouts in a given batch, the model continues to explore stochastically. Temperature-driven sampling means different failure modes surface across steps. The model can eventually find a non-zero-reward rollout through chance alone — this is how standard GRPO eventually cracks hard problems, albeit slowly.

Our method's contribution in this regime is **acceleration**, not enablement. We don't make unsolvable problems solvable (that depends on the conditioned distribution having support over correct strategies). We make the search faster by narrowing the exploration space away from repeatedly-tried failure modes, increasing the probability per step that a correct strategy is sampled.

The more precise claim: on problems where the model has latent capability (non-zero pass@k at large k, whether conditioned or unconditioned), failure conditioning reduces the number of training steps needed to discover and reinforce that capability. The magnitude of this acceleration depends on how concentrated the model's probability mass is on a small number of dominant failure modes — the more concentrated, the more our conditioning helps by suppressing those modes.

On problems where the model genuinely lacks the capability even under conditioning, our method adds overhead (summariser compute, context tokens) without benefit. The annealing mechanism limits this waste: if conditioning never leads to improved solve rates, p_condition remains at 1 but the GRPO gradient on that prompt remains zero regardless, and the overhead is bounded to the summariser cost on persistently-hard prompts.

---

## 4. The Method

### 4.1 Overview

The method modifies the rollout generation phase of standard GRPO training using an in-step two-phase structure following the pattern validated by iGRPO. For prompts where the model's solve rate is below a threshold, a first phase of rollouts ("scouts") are generated, summarised by a lightweight two-level summariser, and the summary is prepended to a second phase of conditioned rollouts. The GRPO advantage computation and gradient update are entirely standard, applied to the conditioned rollouts only.

### 4.2 Training Step (per prompt)

**Given:** Prompt x, rollout budget N per phase (1:1 ratio between scouts and conditioned), solve-rate threshold τ (e.g., 0.25), annealing probability p_condition based on rolling solve rate.

```
1. Check annealing probability:
   - Compute p_condition = max(0, 1 - (solve_rate_x / τ))
   - Flip coin with probability p_condition
   - If tails (no conditioning): generate N rollouts unconditioned → standard
     GRPO → done

2. Scout phase:
   - Generate N rollouts from π_θ(y|x) (unconditioned)
   - Compute rewards for scouts

3. Summarise (synchronous, in-step):
   - Feed each failed scout rollout to the summariser
   - Summariser produces one descriptive summary per failed rollout
   - Feed those rollout summaries to an aggregate summary step
   - Aggregator produces a single narrative summary (8-12 sentences) of
     approaches tried and how they ended up
   - Wrap summary in a randomly selected framing template

4. Conditioned rollouts:
   - Generate N rollouts from π_θ(y|x, c) where c is the failure context block
   - Compute rewards

5. GRPO update:
   - Compute advantages across the N conditioned rollouts ONLY
   - Scouts are excluded from the gradient computation (they served as
     diagnostic input for the summariser)
   - Apply a standard GRPO-style policy gradient with clipping; KL is optional
     and is disabled in the stage-1 PoC (`beta = 0.0`)
   - Update rolling solve rate for prompt x
```

**Note on scout exclusion from gradient:** This is a deliberate hard-problem tradeoff, not a claim of universal superiority. On prompts where scouts are overwhelmingly failures, their advantages are near-degenerate and their gradient contribution is usually negligible. Excluding them concentrates gradient budget on the conditioned rollouts, where informative discoveries are more likely. The cost is that we do not get the implicit anti-dependence benefit of joint unconditioned+conditioned gradient on those prompts. Annealing therefore does essential work in our method: it is the mechanism that reintroduces unconditioned training signal once the prompt begins to crack. We test this tradeoff directly with a scout-excluded vs. scout-included ablation.

**Stage-1 note:** In the proof-of-concept stage, we do **not** use scout-success gating. Once a prompt is routed into the RETRO-GRPO path, it proceeds through summarisation and conditioned rollout generation without an additional early-exit shortcut. That keeps the first experiment focused on testing the conditioned mechanism directly before introducing routing heuristics.

### 4.3 The Summariser

The summariser produces a factual narrative of what the failed attempts did. It does NOT diagnose errors, evaluate correctness, or suggest alternatives. It describes approaches and outcomes.

**First-pass summariser prompt (per failed scout):**

```
Summarise the approach taken in this failed solution attempt in
5-8 sentences. Describe what was tried and how the attempt ended up.
Do not evaluate whether the approach was correct or incorrect. Do not
suggest alternatives.

## Problem
{problem_text}

## Failed Attempt
{failed_scout_rollout}

Summary of approach tried:
```

**Aggregate summariser prompt (over rollout summaries):**

```
Summarise the approaches taken across these failed solution-attempt
summaries in 8-12 sentences. Describe what was tried and how the attempts
ended up. Do not evaluate whether any approach was correct or incorrect.
Do not suggest alternatives.

## Problem
{problem_text}

## Failed Attempt Summaries
{failed_scout_rollout_summaries}

Summary of approaches tried:
```

**Example output (math):** "Most attempts set up the integral in polar coordinates and applied u-substitution on the inner integral, with slight variations in the choice of substitution variable. One attempt converted to Cartesian coordinates instead but encountered difficulties with the resulting bound expressions. All approaches ultimately arrived at expressions that did not simplify to a clean answer."

**Example output (agentic, τ-bench):** "The agent consistently retrieved the order details first, then attempted to process a refund immediately. Most attempts skipped the return eligibility check entirely. One attempt did call check_return_policy but proceeded with the refund before the response was incorporated into the decision. All attempts ended with the refund either rejected by the system or applied incorrectly."

**Scaling to large k:** The default scaling path is hierarchical summarisation: summarise each failed rollout first, then aggregate those rollout summaries. Subset sampling is fallback-only if a single rollout or an unusually large summary set still exceeds the summariser's context window.

**Model choice — configurable with default:**

| Setting | `training_policy` (default) | `frozen_base` |
|---------|---------------------------|---------------|
| Implementation | Same model, same weights, LoRA adapters ON | Same model, LoRA adapters OFF |
| Rationale | Distributional alignment — model reads text from its own "voice." Improving domain understanding produces progressively more specific summaries as training progresses. Precedent: iGRPO and Critique-GRPO's self-critiquing variants both use the training policy in dual roles. | Guaranteed stability — summarisation behaviour is constant throughout training. No interaction between training signal and summarisation quality. |
| Risk | Summarisation style may drift over training (mitigated: summaries are ephemeral, consumed in-step, never accumulated) | Summaries may lack domain specificity that the training policy has developed (mitigated: summarisation is simple enough that base capability suffices) |

Default is `training_policy`. The summarisation task (factual description of a trajectory) is orthogonal to the evaluation task (solving the problem). As the training policy improves at the domain, it becomes more specific in describing what was tried, without improving at implicitly evaluating correctness — because the summary contains no evaluation. This improving specificity is a benefit, not a risk.

### 4.4 Failure Context Block

The summariser's output is wrapped in light, randomised framing and prepended to the problem prompt.

**Format (randomly selected per prompt per step from 4-5 variants):**

```
Variant 1:
[Previous attempts on this problem did not succeed. {summariser_output}]

Variant 2:
[Note: earlier solution attempts for this problem were tried.
{summariser_output} None of these succeeded.]

Variant 3:
[The following approaches were tried on this problem and did not produce
a correct result. {summariser_output}]

Variant 4:
[Prior approaches to this problem: {summariser_output} These did not
lead to a successful outcome.]
```

**Placement:** Prepended to the problem prompt, separated by a newline:

```
{failure_context_block}

{original_problem_prompt}
```

The model reads the failure context before it encounters the problem, so the context influences strategy selection from token 1.

**Framing randomisation** prevents the model from learning a brittle association between a specific token sequence and the "conditioning is present" signal, which would make annealing more disruptive.

### 4.5 Annealing

**Role of annealing:** Annealing remains the mechanism that transfers gains from the more capable conditioned policy back into the unconditioned policy. That conceptual role is unchanged.

**However, it is no longer the first experiment.** The experimental program is staged:

1. **Stage 1:** validate uplift from descriptive failure-conditioning with **no annealing**.
2. **Stage 2:** add **batch-consistent annealing** using a global EMA-governed conditioning rate.
3. **Stage 3:** add the **per-prompt coin-flip** annealing variant.

This ordering is deliberate. There is no value in optimising annealing before showing that summary-conditioned rollouts improve training at all.

**Full-method schedule (research default):** Performance-gated appearance annealing remains the intended long-horizon mechanism once the base effect is validated.

```
p_condition(x) = max(0, 1 - (solve_rate(x) / τ))
```

Where solve_rate(x) is a rolling average over the last K appearances of prompt x and τ is the solve-rate threshold.

**Stage-2 operational schedule:** use a **global EMA of performance** to set the conditioning rate, and apply that decision with **batch consistency**. Either the whole batch uses the conditioning path or the whole batch runs standard GRPO.

**Stage-3 operational schedule:** keep the global EMA-governed conditioning rate, but make the decision at the prompt level via a coin flip. This tests whether mixed conditioned/unconditioned prompts within the same batch improve internalisation enough to justify the extra throughput cost.

We still prefer **appearance annealing** over detail annealing. We do not anneal summary content itself.

### 4.6 Anti-Dependence Mechanisms

Four defences against the model developing conditioning dependence:

1. **Framing variation.** Randomised wrappers prevent brittle token-level associations.
2. **Inter-batch appearance annealing.** As solve rate increases, the model encounters the prompt unconditioned more frequently. It learns to succeed without conditioning before conditioning is fully removed.
3. **No solution leakage.** Summaries are descriptive, not prescriptive. No ground truth, no reward signal, no diagnostic judgment enters the context.
4. **Soft framing engages meta-reasoning.** The transfer mechanism is meta-cognitive (learning to reason about strategy selection) rather than instruction-following (learning to obey avoidance commands). Meta-cognitive skills persist without context triggers.

**Monitoring:** Track the gap between conditioned and unconditioned solve rates on prompts currently receiving conditioning. A growing gap indicates increasing dependence. A shrinking gap indicates successful internalisation.

### 4.7 Configuration: Post-Step Alternative

As an ablation, we test a post-step summarisation variant that eliminates the scout overhead:

```
1. Generate all N rollouts unconditioned
2. Compute rewards, compute advantages, apply standard GRPO update
3. Synchronously after the gradient update: for prompts below the solve-rate
   threshold, run the summariser over the failed rollouts from this step
4. Store the summary for this prompt
5. Next time this prompt appears in a batch: prepend the stored summary
   (subject to annealing coin flip)
```

**Advantages:** Zero scout overhead — all N rollouts contribute to the GRPO gradient every step. Simpler pipeline — no two-phase rollout generation.

**Disadvantage:** The summary is one-appearance stale. Between generation and use, the model has trained on other prompts, and transfer from related tasks may have shifted the failure modes on this prompt.

**Prediction:** Post-step is likely equivalent to in-step on math benchmarks (slow transfer, stable failure modes on hard problems) and potentially inferior on agentic benchmarks (faster transfer, shifting failure frontiers as the model reaches new capability levels exposing new failure modes).

This ablation is particularly informative on τ-bench, where the agentic transfer dynamics test whether staleness matters in practice.

---

## 5. Benchmarks

### 5.1 Research-Level Benchmark Positioning

For full-scale evaluation and paper positioning, the method remains naturally relevant to hard mathematical reasoning benchmarks and later to hard agentic benchmarks.

Mathematical benchmarks of interest remain:
- AIME-style competition math,
- OlymMATH-HARD,
- HMMT-style hard competition sets,
- MATH-style benchmark suites.

Agentic benchmarks such as τ-bench remain a natural later-stage extension once the core math mechanism is validated.

### 5.2 Proof-of-Concept Benchmark Choice

For the compute-constrained PoC, the benchmark plan is deliberately narrower.

**Stage 1 uses a single benchmark: MATH-500.**

Why:
- it is hard enough for conditioning to matter,
- it is much more stable than tiny extreme-hard sets when used as the only benchmark,
- it is simple to score deterministically,
- it gives a cleaner first answer to the question "does the method help training on hard math at all?"

**Reporting for stage 1:**
- overall MATH-500,
- plus a headline hard slice within MATH-500.

GSM8K is not part of the core stage-1 evidence package, and AIME is not used as the sole first benchmark because it is too small and noisy for the first proof experiment.

### 5.3 Evaluation Metrics

The primary early metric remains the learning curve on hard problems.

For the PoC, that means:
- solve rate vs training step on MATH-500,
- hard-slice solve rate vs training step,
- pass@1 and pass@4 where budget permits.

Deterministic answer verification is the default evaluation path; no LLM judge is required for the stage-1 math setup.

## 6. Training Configuration

### 6.1 Research Configuration vs PoC Configuration

The original research configuration assumes larger benchmark suites, larger models, and more extensive ablations. The PoC configuration is intentionally narrower and budget-driven.

### 6.2 PoC Default Configuration

**Model and stack**
- Qwen3-8B
- bf16 LoRA via a custom `transformers + peft + accelerate` stack
- LoRA rank 32
- L4 as the default Modal GPU
- Qwen3 rollouts use thinking mode; summary generation disables thinking
- custom GRPO loss module with default `scale_rewards = "group"` and `loss_type = "dapo"`

**Training data**
- DeepMath-103K
- hard-filtered subset for stage 1
- exact stage-1 subset size: 1,200 problems
- filtering target: keep prompts in the hard-problem regime, operationally around pass@4 ≤ 0.25

**Benchmark**
- stage 1: MATH-500 only

**Generation and batching**
- per-device prompt slots: 3
- gradient accumulation steps: 4
- rollouts per phase: N = 4
- max completion length: 1024 tokens
- effective prompt batch size: 12
- effective conditioned completions per optimizer step: 48
- target unique prompts per generation cycle: 3
- stage-1 optimizer steps: 300 (1,200 problems × 3 epochs / batch 12)

**Pipeline**
- in-step two-phase generation
- training-policy hierarchical summariser
- framing randomisation retained
- no annealing in stage 1

### 6.3 Stage-Specific Defaults

**Stage 1**
- baseline GRPO vs RETRO-GRPO
- no annealing
- hard-filtered broad training subset
- objective: validate uplift from descriptive failure-conditioning

**Stage 2**
- add batch-consistent annealing via global EMA

**Stage 3**
- add per-prompt coin-flip annealing

### 6.4 Operational Note on Cost

Budget planning must account for GPU, CPU, and memory costs together. Because actual throughput depends heavily on rollout lengths and the fraction of prompts taking the conditioned branch, wall-clock projections should be treated as smoke-test validated rather than fixed analytically.

## 7. Implementation Complexity

The implementation burden depends on the stage.

### 7.1 Stage 1 Complexity

Stage 1 is intentionally simpler than the full method:
- no annealing state,
- no per-prompt solve-rate tracker,
- no EMA schedule,
- no coin flips.

The added logic is:
- batched scout generation,
- deterministic scout scoring,
- summarisation for every prompt routed into the RETRO-GRPO path using that prompt's failed scouts,
- batched conditioned rollout generation for those prompts,
- batch assembly before the local GRPO update,
- modular reward scaling and loss reduction so GRPO, DAPO, and Dr.GRPO stay swappable.

This is the minimal implementation needed to answer the first-order mechanism question.

### 7.2 Later-Stage Complexity

Stage 2 adds a global EMA-governed annealing schedule with batch consistency.
Stage 3 adds per-prompt routing on top of the global schedule.

This staged implementation path is intentional: it keeps the first experiment simple and only adds schedule-management logic after the base uplift is established.

## 8. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Model develops conditioning dependence | Medium | Appearance annealing gated on performance, framing variation, monitoring conditioned/unconditioned gap |
| Summariser quality drifts during training (training_policy mode) | Low | Summarisation is factual description, not diagnosis; summaries are ephemeral and consumed in-step; frozen_base mode available as fallback |
| Conditioning overhead exceeds benefit on easy problems | N/A | Conditioning only triggers below solve-rate threshold; easy problems are never conditioned |
| Method shows no improvement over strong baselines | Medium | Focus evaluation on hard-problem regime; ablations isolate which components contribute |
| Transfer from conditioned to unconditioned is weak | Medium | Soft framing maximises transfer by training meta-reasoning; annealing forces autonomous performance |
| Scout rollouts are wasted compute | Low | Scouts on hard problems would produce near-zero gradient anyway; they're repurposed as diagnostic input |

---

## 9. Paper Positioning

**Title direction:** "Failure-Conditioned Exploration for Hard-Problem RL in LLMs" or "Learning What Not to Try: Descriptive Failure Context for RL Exploration in Language Models"

**Core claim:** On hard problems where RL repeatedly samples the same unsuccessful strategies, telling the model what it already tried — in plain descriptive language, before it generates again — makes it find rewarding solutions faster than gradient alone, and the gains transfer after the context is withdrawn. This is a sample-efficiency claim, not a capability-creation claim.

**Contribution framing:**

The in-step two-phase pipeline structure is adopted from iGRPO and is not a structural contribution. Our contributions are:

1. **Information type:** We replace positive self-conditioning or per-response critique with cross-rollout descriptive failure narratives ("here is what was tried across failed attempts"). The novelty is the information content, not the existence of a second phase.

2. **Mechanism:** We show how routing failure information through the context window changes search without changing GRPO itself. The guidance is semantic and pre-generative: it acts on strategy selection before sampling, rather than suppressing failed token sequences through gradient or correcting one failed trajectory after the fact.

3. **Learning-dynamics contribution:** We make explicit the tradeoff between concentrating gradient on exploration-improved samples and preserving unconditioned practice. Scout exclusion sharpens the hard-problem learning signal; performance-gated annealing is the mechanism that transfers the gains back to the unconditioned policy.

4. **Difficulty-regime positioning:** We argue for a cleaner division of labour across methods. Positive conditioning is strongest when a good draft already exists, critique-based refinement is strongest when a specific trajectory mostly needs correction, and failure-conditioned exploration is strongest when the policy is repeatedly spending rollout budget on the same losing strategy classes.

**Why this matters beyond the benchmark result:** More broadly, the method probes whether some learning-relevant information is better routed through the model's semantic in-context channel than through weight updates alone. We do not claim a general memory system or a new form of continual learning. The grounded claim is narrower: strategic information about what has already been tried and failed may be a higher-bandwidth signal for search than gradient alone, and RL can then consolidate the benefit back into the base policy.

**Related work positioning:** The method synthesises ideas from ICRL (annealing in-context conditioning during RL), SDFT (distilling conditioned behaviour into weights), NSR (negative signal preserves exploration), and iGRPO (within-step conditioning for RL), while contributing a distinct information type and mechanism that addresses a regime where these existing methods are weakest.
