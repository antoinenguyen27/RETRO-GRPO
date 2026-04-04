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

The method modifies the rollout generation phase of standard GRPO training using an in-step two-phase structure following the pattern validated by iGRPO. For prompts where the model's solve rate is below a threshold, a first phase of rollouts ("scouts") are generated, summarised by a lightweight summariser, and the summary is prepended to a second phase of conditioned rollouts. The GRPO advantage computation and gradient update are entirely standard, applied to the conditioned rollouts only.

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

3. Assess:
   - If scout solve rate > τ: skip conditioning, apply standard GRPO to
     scout rollouts directly (they become the training rollouts) → done
   - If scout solve rate ≤ τ: trigger summarisation

4. Summarise (synchronous, in-step):
   - Feed all failed scout rollouts to the summariser in a single call
   - Summariser produces a single narrative summary (3-5 sentences) of
     approaches tried and how they ended up
   - Wrap summary in a randomly selected framing template

5. Conditioned rollouts:
   - Generate N rollouts from π_θ(y|x, c) where c is the failure context block
   - Compute rewards

6. GRPO update:
   - Compute advantages across the N conditioned rollouts ONLY
   - Scouts are excluded from the gradient computation (they served as
     diagnostic input for the summariser)
   - Apply standard GRPO policy gradient with clipping and KL penalty
   - Update rolling solve rate for prompt x
```

**Note on scout exclusion from gradient:** This is a deliberate hard-problem tradeoff, not a claim of universal superiority. On prompts where scouts are overwhelmingly failures, their advantages are near-degenerate and their gradient contribution is usually negligible. Excluding them concentrates gradient budget on the conditioned rollouts, where informative discoveries are more likely. The cost is that we do not get the implicit anti-dependence benefit of joint unconditioned+conditioned gradient on those prompts. Annealing therefore does essential work in our method: it is the mechanism that reintroduces unconditioned training signal once the prompt begins to crack. We test this tradeoff directly with a scout-excluded vs. scout-included ablation.

**Note on step 3 early exit:** If the scouts reveal the prompt is above threshold (the model has improved since the solve rate was last computed), we skip conditioning entirely and use the scouts as standard GRPO training rollouts. No compute is wasted — the scouts serve double duty as both assessment and training data.

### 4.3 The Summariser

The summariser produces a factual narrative of what the failed attempts did. It does NOT diagnose errors, evaluate correctness, or suggest alternatives. It describes approaches and outcomes.

**Summariser prompt (single call over all failed scouts):**

```
Summarise the approaches taken across these failed solution attempts in
3-5 sentences. Describe what was tried and how the attempts ended up.
Do not evaluate whether any approach was correct or incorrect. Do not
suggest alternatives.

## Problem
{problem_text}

## Failed Attempts
{all_failed_scout_rollouts}

Summary of approaches tried:
```

**Example output (math):** "Most attempts set up the integral in polar coordinates and applied u-substitution on the inner integral, with slight variations in the choice of substitution variable. One attempt converted to Cartesian coordinates instead but encountered difficulties with the resulting bound expressions. All approaches ultimately arrived at expressions that did not simplify to a clean answer."

**Example output (agentic, τ-bench):** "The agent consistently retrieved the order details first, then attempted to process a refund immediately. Most attempts skipped the return eligibility check entirely. One attempt did call check_return_policy but proceeded with the refund before the response was incorporated into the decision. All attempts ended with the refund either rejected by the system or applied incorrectly."

**Scaling to large k:** When N is large enough that all scout rollouts exceed the summariser's context window, sample a subset of 16-20 rollouts selected mechanically for diversity (longest, shortest, and random from the middle). The summariser still produces the same fixed-size 3-5 sentence output. The sampling is a pre-filter for context limits, not a judgment call.

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

**Role of annealing:** Annealing is not merely a graceful way to withdraw a training aid. In our default design, when conditioning triggers, the prompt receives gradient only through conditioned rollouts. On persistently hard prompts, that means the model may go many updates without directly practising unconditioned generation on that prompt at all. Annealing is therefore the mechanism that transitions learning back into the unconditioned distribution. It is structurally necessary, not optional polish.

**Mechanism:** Appearance probability annealing (the conditioning either appears in full or not at all). We do not anneal summary detail, which would introduce a novel distribution of conditioning text mid-training and add noise.

**Schedule:** Per-prompt, self-calibrating, gated on performance.

```
p_condition(x) = max(0, 1 - (solve_rate(x) / τ))
```

Where solve_rate(x) is a rolling average over the last K appearances of prompt x (K ≈ 5-10). τ is the solve-rate threshold (hyperparameter, suggested 0.25-0.5).

When solve_rate = 0 (never solved): p_condition = 1 (always conditioned).
When solve_rate = τ (solving at target rate): p_condition = 0 (conditioning fully withdrawn).
Linear interpolation between.

**Key property: performance-gated, not step-gated.** The conditioning stays at full strength as long as the model is struggling on a prompt. It only recedes when the model demonstrates actual improvement on that specific prompt. This means:

- A prompt that remains hard throughout training keeps full conditioning throughout.
- A prompt that cracks quickly begins receiving unconditioned practice quickly.
- There is no global schedule to tune. The per-prompt solve rate is the only signal.
- The schedule explicitly manages the return path from a more capable conditioned policy to the unconditioned policy we care about at inference time.

Another way to view the schedule is as a controlled handoff. Early in training, the conditioned policy is allowed to do the exploration-heavy work. As soon as the prompt becomes tractable, the annealing coin flips begin reintroducing unconditioned rollouts, so the model must demonstrate that the improvement has transferred into its weights rather than remaining contingent on the context block.

**Experimental knob:** Detail annealing (instructing the summariser to produce shorter/vaguer summaries as solve rate improves) is a secondary axis that can be explored but is NOT the default. Appearance annealing is cleaner and avoids managing multiple summariser prompt variants.

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

### 5.1 Primary: Mathematical Reasoning (Hard)

**AIME 2025** (30 problems, olympiad-level, integer answers 000-999). The standard comparison point. Every paper in this space reports it (iGRPO, Critique-GRPO, POPE, PrefixRL, NSR). Required for positioning.

**OlymMATH-HARD** (olympiad-level, specifically designed to resist shortcut strategies). Even frontier models achieve only 30-60%. For 7B models under RL training, the majority of these problems will be in the low-solve-rate regime where our method has maximum leverage.

**HMMT 2025** (competition mathematics). Used by POPE specifically because it's in the zero-reward regime for most models. Direct comparison point for the hard-problem exploration argument.

### 5.2 Secondary: Agentic (Hard, Unsaturated)

**τ-bench retail** (multi-turn customer service with API tool calls and policy guidelines). GPT-4o succeeds on <50% of tasks with pass^8 <25%. Key properties that make it suitable:

- Failure modes are structured and recurring (wrong tool, policy violation, missing verification). Naturally summarisable.
- The pass^k metric directly tests the exploration-preservation argument.
- Far from saturated — large room for improvement.
- Verifiable evaluation via database state comparison (no LLM judge needed for reward).

**Adaptation for agentic setting:** A "rollout" is a full multi-turn conversation. The summariser describes the sequence of actions taken across failed conversations. The failure context block is prepended to the agent's system prompt before the conversation begins. The GRPO update operates on the full trajectory reward.

The in-step vs post-step ablation is particularly informative on τ-bench due to faster transfer dynamics and combinatorially expanding failure frontiers as the model reaches new capability levels.

### 5.3 Evaluation Metrics

**Pass@k curves** (k = 1, 4, 16, 64, 256). The primary evidence. Pass@1 demonstrates convergence speed. Pass@k at large k demonstrates exploration preservation. The method should improve pass@1 without degrading pass@k relative to baselines.

**Learning curves.** Plot solve rate vs. training step for hard problems specifically (problems where the base model has pass@128 < 0.1). This shows the acceleration effect directly.

**Conditioned vs. unconditioned gap over training.** Demonstrates that the model is internalising the failure-avoidance, not depending on context. This gap should shrink over training.

### 5.4 Baselines

1. **Standard GRPO** (primary baseline — full positive and negative advantage)
2. **iGRPO** (positive self-conditioning via best draft)
3. **Critique-GRPO** (natural language critique + refinement)
4. **NSR-weighted REINFORCE** (upweighted negative sample reinforcement)
5. **GRPO + our method** (the proposed method)

Ablation conditions:
- Hard prescriptive framing vs. soft descriptive framing (isolates framing contribution)
- In-step vs. post-step summarisation (isolates freshness contribution)
- Training policy summariser vs. frozen base summariser (isolates distributional alignment contribution)
- Scout-excluded vs. scout-included gradient (tests whether joint gradient materially improves transfer or reduces dependence on medium-difficulty prompts)
- PSR-only + our method (full decoupling — no negative gradient, all failure signal through context)
- Detail annealing vs. appearance annealing

### 5.5 Base Models

**Qwen2.5-7B-Instruct** or **Qwen3-8B** to match iGRPO and Critique-GRPO baselines. These are the standard models in the current literature and enable direct comparison.

---

## 6. Training Configuration

### 6.1 GRPO Hyperparameters

Follow standard settings from the baseline papers:
- Rollouts per phase: N = 8 (8 scouts, 8 conditioned when conditioning triggers)
- Temperature: 1.0 for rollout generation
- Learning rate: 1e-6
- KL penalty coefficient: 0.001
- Clipping epsilon: 0.2
- Max response length: 2048 tokens (math), 4096 tokens (agentic)
- Batch size: 64-256 prompts per step

### 6.2 Conditioning Hyperparameters

- Solve-rate threshold τ: 0.25 (suggested, tune on validation)
- Rolling window for solve rate K: 5 appearances
- Scout-to-conditioned ratio: 1:1 (N scouts, N conditioned)
- Summariser mode: `training_policy` (default) or `frozen_base` (ablation)
- Pipeline mode: `in_step` (default) or `post_step` (ablation)
- Context block framing variants: 4-5 (randomly selected per prompt per step)

**Note on total rollout budget:** When conditioning triggers, the total per-prompt rollout count is 2N (N scouts + N conditioned). When conditioning does not trigger (prompt above threshold or annealing coin flip says skip), the total is N (standard GRPO). The overhead from 2N rollouts applies only to prompts below the solve-rate threshold, and the fraction of such prompts decreases over training as the model improves.

### 6.3 Training Data

**Math:** OpenR1-Math-220k subsets (4k-32k prompts), matching Critique-GRPO's setup for direct comparison.

**Agentic:** τ-bench task set with programmatic task generation for training scale.

---

## 7. Implementation Complexity

The method adds to a standard GRPO training loop:
- A per-prompt solve-rate tracker (dictionary: prompt_id → list of recent solve rates)
- A coin flip per prompt per batch (one line)
- A two-phase rollout generation for conditioned prompts (following iGRPO's pattern)
- A summariser call between phases (single inference call, same model, same GPU)
- A string concatenation and prepend (trivial)

No changes to the GRPO algorithm, advantage computation, gradient update, or training infrastructure. The summariser uses the training policy (default) or base model with LoRA adapters toggled — either way, it's the same model already loaded on the same GPU. No additional model loading, no external API calls.

Total additional code beyond standard GRPO: approximately 200-300 lines. The two-phase rollout structure follows iGRPO's pattern, which is already implemented in frameworks like VeRL.

---

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
