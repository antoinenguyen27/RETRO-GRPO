# Refreshed Failure Memory for Reinforcement Learning in LLMs

## Research Specification Document - v1

---

## 1. Core Idea

The shallow RETRO-GRPO design uses same-step failure context: generate scout rollouts, summarize the failed attempts, prepend that description to the prompt, then generate a second conditioned rollout phase. That design is strong because the information is fresh, but it only has **depth 1**. It knows what failed in the current appearance of the prompt and nothing about what failed the last time the prompt was seen.

This document proposes the best practical extension: **refreshed per-prompt episodic failure memory**.

The central idea is simple:

1. retrieve a compact memory for the current prompt if one exists,
2. run fresh unconditioned scouts,
3. summarize the current failed scouts,
4. merge the stored memory with the fresh summary into an updated memory,
5. condition the second rollout phase on that refreshed memory,
6. train on the conditioned rollouts only,
7. update the prompt-local memory state and solve-rate statistics.

This design adds **memory depth** without giving up the strongest property of shallow RETRO-GRPO: the second rollout phase is still anchored to what the current policy just failed to do.

The claim is intentionally narrow. We do **not** claim a general memory architecture, a new form of continual learning, or an agent-style long-horizon retrieval system. The object being stored is much smaller and much more specific:

- prompt-local,
- failure-focused,
- compressed into strategy-level text,
- refreshed by current evidence before use,
- retired once the prompt exits the hard-problem regime.

The guiding hypothesis is that strategic information about what has already been tried and failed is useful not only within one training step, but across multiple appearances of the same prompt. The key practical constraint is that this information cannot be allowed to drift too far away from the current policy frontier. That is why the default design is **refreshed memory**, not stale memory replay and not teacher-student distillation first.

---

## 2. Why We Believe This Works

### 2.1 The Foundational Mechanism

The shallow RETRO-GRPO argument already identifies an important separation: failure information enters through the **context window**, while learning still happens through standard RL updates. That separation remains intact here.

The difference is that the context is no longer purely ephemeral. Instead of conditioning only on "what failed right now," the model conditions on "what has repeatedly failed for this prompt, refreshed by what failed right now." This should matter whenever prompt appearances are sparse enough that the same hard problem is encountered multiple times over training, yet structured enough that failure modes recur across those appearances.

The refreshed-memory design therefore combines two benefits:

- **Cross-appearance accumulation.** The memory carries forward information about recurring failure modes across prompt visits.
- **Current-step re-anchoring.** Fresh scouts stop the memory from becoming detached from the current policy's actual error surface.

The practical thesis is that these two benefits are complementary. Accumulation without re-anchoring becomes stale. Re-anchoring without accumulation collapses back to shallow RETRO-GRPO.

### 2.2 Why This Is Better Than Pure Stale Memory

The simplest way to add memory depth is to summarize failures after a GRPO step and save the summary for the next appearance of the prompt. That is attractive because it avoids the in-step two-phase refresh path. But used by itself, it has a structural weakness: the memory is already old when it is consumed.

Between the moment a summary is written and the next time the prompt appears:

- the policy has changed,
- related prompts may have transferred useful skills,
- some prior failure modes may no longer dominate,
- new failure frontiers may have appeared.

Pure stale memory therefore risks conditioning on yesterday's frontier. That can still help when failure modes are extremely stable, but it is not the best practical default.

Refreshed memory fixes this by letting old memory act as a prior, not as a frozen instruction. The fresh scout phase tells the system whether the old memory still matches the current policy. If it does, the memory is reinforced. If it does not, the memory is revised or retired.

### 2.3 Why Per-Prompt Memory Is Better Than Global Memory

The failure information we care about is highly prompt-specific. "Tried polar coordinates and got stuck in substitution" is useful for one integral and irrelevant for another. A global memory store of generic failure summaries would quickly blur distinct strategy classes together and create weak, noisy conditioning.

Per-prompt memory has three advantages:

- it preserves causal relevance,
- it avoids leaking unrelated failure stories into the prompt,
- it makes retirement and freshness management tractable.

The design is therefore not a semantic search memory over the entire training set. It is a prompt-keyed episodic record attached only to the prompt currently being trained.

### 2.4 Why Compressed Summaries Beat Raw Trajectory Storage

Long-term storage of raw scout rollouts is unattractive for both algorithmic and practical reasons.

Algorithmically:

- raw text encourages token-level anchoring instead of strategy-level abstraction,
- it makes the model overfit to specific wording rather than the class of failed approaches,
- it increases the chance that stale details dominate.

Practically:

- raw trajectories are expensive to store and re-inject,
- token budgets get consumed by historical clutter,
- prompt lengths become unstable.

Compressed descriptive summaries are better because they retain the strategic content while discarding token-level noise. This is the same reason the shallow RETRO design prefers factual summary blocks over raw failed rollouts.

### 2.5 Why Merge-Refresh Beats Append-Only Accumulation

An append-only memory log is the most natural first idea and the wrong default.

If the system continually appends failure summaries across appearances, the memory grows in length while decreasing in precision:

- repeated failure modes are duplicated,
- outdated failure modes linger,
- contradictory summaries accumulate,
- the conditioning block drifts from a compact search prior into a noisy archive.

The better design is **merge-refresh**:

- old memory provides prior structure,
- current failures provide corrective evidence,
- the output is a new compact memory with fixed budget,
- the new memory replaces the old record.

This turns memory into a tracked latent state for the prompt rather than an ever-growing log.

### 2.6 Why Teacher-Student Is Not the Best First Move

Teacher-student training is a real option: let a memory-conditioned teacher produce a stronger policy distribution, then train a memory-free student to match it. In principle this addresses conditioning dependence more directly than ordinary annealing.

But as the default practical design it is weaker for three reasons.

First, it adds a second mechanism before the first mechanism is validated. If memory depth does not help the rollout policy on its own, teacher-student layering does not solve the underlying problem.

Second, it increases complexity sharply:

- another policy role,
- another objective,
- more instability in distribution matching,
- more compute,
- less direct attribution when the result changes.

Third, it changes the research question. The primary question should be whether refreshed failure memory improves exploration across prompt appearances. Distillation is a downstream transfer mechanism, not the cleanest first test of that hypothesis.

The right ordering is therefore:

1. validate memory depth as a direct rollout-conditioning mechanism,
2. add appearance annealing and dependence monitoring,
3. only then test teacher-student transfer as a later-stage extension.

---

## 3. What Is Novel Here

### 3.1 Relative to Shallow RETRO-GRPO

The novelty over shallow RETRO-GRPO is not "conditioning exists" and not "a second rollout phase exists." Those are inherited.

The new contribution is:

- persistent but bounded prompt-local failure memory,
- refreshed before use by current scout evidence,
- merged rather than appended,
- explicitly managed with decay and retirement,
- designed to improve exploration across prompt appearances without surrendering same-step freshness.

Shallow RETRO-GRPO is an in-step search redirection method. Long RETRO-GRPO adds a prompt-local memory state so that search redirection can exploit repeated evidence over time.

### 3.2 Relative to Pure Post-Step Memory

Pure post-step memory is more novel than shallow RETRO-GRPO in the narrow sense that it introduces cross-appearance persistence. But it is not the best design because the stored signal is consumed later without revalidation.

The innovation here is the claim that **memory depth should be refreshed at the moment of use**. Persistence alone is not enough. The practical contribution is the combination of:

- carryover from prior appearances,
- current-step scout verification,
- compact state replacement.

### 3.3 Relative to Teacher-Student Distillation

Teacher-student training is a known pattern. Using a memory-conditioned teacher is an interesting application of that pattern, but it is not the clearest or most defensible main contribution.

The cleaner claim is narrower:

> A prompt-local failure memory, refreshed by current scout evidence before use, improves exploration more effectively than either ephemeral same-step memory alone or stale persistent memory alone.

Teacher-student training may later become the best way to consolidate this capability into a memory-free deployment policy, but it should be framed as a later transfer layer, not the primary design novelty.

### 3.4 Relative to Other RL Prompt-Conditioning Methods

The positioning against nearby methods remains similar to shallow RETRO-GRPO.

Against positive self-conditioning methods such as iGRPO, the memory acts as a repeller from known-bad regions rather than an attractor toward a best draft.

Against critique-based methods, the memory is not a diagnosis of one failed trajectory. It is a compact cross-appearance description of strategy classes that keep failing for this prompt.

Against negative-gradient methods such as NSR, the suppression still happens through context rather than direct negative weight updates. The model reads what tends not to work and decides what to avoid semantically.

The new difference is temporal:

- shallow RETRO aggregates across rollouts within one step,
- long RETRO aggregates across rollouts **and** across prompt appearances,
- but always through a refreshed compact state rather than an accumulating archive.

---

## 4. The Method

### 4.1 Overview

The default long RETRO-GRPO pipeline is a three-source conditioning mechanism:

1. the original prompt,
2. a stored prompt-local memory from prior appearances,
3. a fresh current-step summary derived from scout failures.

The stored memory and current-step summary are merged into a single refreshed memory block, and only that refreshed block is used for the conditioned rollout phase.

The design principle is strict:

- do not inject raw old memory directly if current evidence says it is obsolete,
- do not append old and new summaries verbatim,
- do not keep a long historical archive in the prompt.

The policy should see one concise refreshed account of what this prompt's failure frontier currently looks like.

### 4.2 Canonical Training Step

For a prompt `x` in the hard-problem regime:

```text
1. Retrieve stored MemoryRecord(x) if it exists.
2. Generate N scout rollouts from the unconditioned policy.
3. Score scouts with the task reward.
4. Summarize the current failed scouts into CurrentSummary(x).
5. Merge MemoryRecord(x) and CurrentSummary(x) into RefreshedMemory(x).
6. Decide whether RefreshedMemory(x) should be injected this appearance.
7. Generate N conditioned rollouts from pi_theta(y | x, RefreshedMemory(x)).
8. Compute rewards and apply standard GRPO on the conditioned rollouts only.
9. Update solve-rate EMA and memory metadata for x.
10. Decay or retire memory if the prompt exits the hard regime.
```

This is the canonical long RETRO path. It preserves the key shallow RETRO choice that the gradient is applied to the conditioned rollouts only. The scouts remain diagnostic input for the memory updater.

### 4.3 `MemoryRecord`

Each prompt is associated with at most one active memory record.

Conceptually:

```python
MemoryRecord = {
    "prompt_id": str,
    "summary_text": str,
    "failure_modes": list[str],
    "solve_rate_ema": float,
    "last_updated_step": int,
    "age": int,          # or ttl_remaining
    "stability_score": float,
}
```

Field meanings:

- `prompt_id`: stable identifier for the prompt or dataset row.
- `summary_text`: the compact narrative memory injected into the prompt when active.
- `failure_modes`: a short structured list of recurring strategy classes or failure patterns.
- `solve_rate_ema`: prompt-local moving estimate of how often the current policy solves the prompt.
- `last_updated_step`: optimizer step of the most recent memory refresh.
- `age` or `ttl`: freshness control.
- `stability_score`: estimate of whether the failure frontier is recurring versus rapidly shifting.

The record should remain small. The goal is not a rich database entry. The goal is a single compact retrieval object that can survive across prompt appearances without overwhelming the context window.

### 4.4 Memory Operators

The memory system is defined by five core operators.

#### `retrieve`

Input:

- `prompt_id`

Output:

- existing `MemoryRecord` if active,
- `None` if no memory exists or the record has been retired.

Default behavior:

- retrieve only prompt-local state,
- do not perform nearest-neighbor retrieval,
- ignore retired records.

#### `summarize_current_failures`

Input:

- prompt text,
- current failed scout rollouts.

Output:

- compact summary of the current failed strategy classes,
- optional short list of extracted `failure_modes`.

Default behavior:

- use descriptive summarization only,
- avoid diagnosis, suggestions, or solution leakage,
- return `None` if there are no current failed scouts.

#### `merge_refresh`

Input:

- existing `MemoryRecord` or `None`,
- current summary or `None`,
- current step metadata.

Output:

- new compact `MemoryRecord`.

Default behavior:

- if only current summary exists, create a new record,
- if only old memory exists and scouts show no failures, decay the record rather than strengthen it,
- if old and new conflict, current evidence wins,
- keep a fixed token budget,
- rewrite the record into one concise refreshed summary rather than concatenating the texts.

This operator is the heart of the method. It decides how cross-appearance persistence is converted into a current-step search prior.

#### `decay`

Input:

- `MemoryRecord`,
- solve-rate and current-step evidence.

Output:

- weakened record with lower priority or shorter remaining life.

Default behavior:

- decay memory when the prompt begins solving reliably,
- decay memory when fresh scouts show no relevant failures,
- decay memory when the record is old and unsupported by recent evidence.

#### `retire`

Input:

- `MemoryRecord`,
- retirement condition.

Output:

- inactive memory.

Default behavior:

- retire when solve-rate EMA crosses the hard-problem threshold comfortably,
- retire when the record reaches TTL without reinforcement,
- retire when the memory becomes inconsistent with repeated fresh evidence.

### 4.5 Memory Lifecycle

The memory should be understood as a small prompt-local state machine rather than as a passive text artifact.

The lifecycle has six phases.

**1. Absent**

The prompt has no active memory record.

This is the default state when:

- the prompt has never accumulated useful failure evidence,
- an older record has already expired,
- or the prompt has been retired from memory because it is no longer in the hard regime.

In this state, the prompt is treated like shallow RETRO-GRPO on first contact: fresh scouts are generated first, and only current evidence can create a memory.

**2. Created**

A new memory record is created when a hard prompt produces failed scouts and there is no usable prior record.

The creation event stores:

- a compact refreshed `summary_text`,
- a short set of extracted `failure_modes`,
- the prompt-local `solve_rate_ema`,
- `last_updated_step`,
- freshness metadata such as `age` or TTL,
- and an initial `stability_score`.

The important design choice is that even the first record is compact. The system does not begin life as a raw failure log and then get compressed later. Compression is the default from the start.

**3. Refreshed**

When the prompt reappears and an active record already exists, the memory is refreshed rather than replayed blindly.

The refresh step is:

1. retrieve the current record,
2. run fresh scouts,
3. summarize current failures,
4. merge old memory with new evidence,
5. rewrite one new compact record.

This is the central phase of the lifecycle. It is where long RETRO differs both from shallow RETRO and from stale post-step memory. The old record survives only by agreeing with current evidence strongly enough to remain useful.

**4. Used or Skipped**

After refresh, the record is eligible for conditioning, but it is not always injected.

There are two possible outcomes for the current appearance:

- **used:** the refreshed memory is prepended to the prompt because the prompt is still in the hard regime and the record is still active,
- **skipped:** the record is not injected because memory dropout fires, the prompt is no longer sufficiently hard, or fresh evidence suggests that negative conditioning is no longer warranted.

This distinction matters. A memory record can still exist while being temporarily skipped for a particular appearance. Existence and injection are not the same thing.

**5. Decayed**

If fresh scouts no longer support the old failure picture, or if the prompt begins solving more reliably, the memory enters a decay phase.

Decay means:

- decreasing the effective priority of the record,
- increasing age,
- weakening confidence in the stored failure modes,
- moving the record closer to retirement if reinforcement does not return.

Decay is what stops the memory from acting like a stale negative prior after the policy frontier has shifted.

**6. Retired**

The record is removed when it is no longer useful.

Retirement happens when:

- the prompt's `solve_rate_ema` rises above the hard-problem threshold,
- TTL expires without meaningful reinforcement,
- or repeated fresh evidence indicates that the stored record is obsolete or contradictory.

Once retired, the prompt returns to the **Absent** state. If it later becomes hard again and produces recurring failures, a new record can be created from scratch.

This full lifecycle gives the memory a narrow operational meaning:

- it is created by repeated failure,
- refreshed by fresh evidence,
- used only while the prompt remains hard,
- weakened when support disappears,
- and retired once it stops being the right search prior.

### 4.6 Memory Management Defaults

The default practical policy is:

- **per-prompt memory**, not global memory,
- **compressed summaries only**, never raw full rollouts as long-term state,
- **fixed token budget** for injected memory,
- **merge-refresh**, not append-only accumulation,
- **recency-weighted updates**,
- **rapid decay or retirement** once the prompt starts solving reliably,
- **occasional memory dropout** and later appearance annealing to avoid dependence.

These defaults are not incidental engineering choices. They define what kind of memory this is. If the system becomes append-only, global, or raw-trajectory based, it ceases to be the intended method and turns into a looser retrieval archive.

### 4.7 Conditioning Policy

The memory is not always injected.

The default conditioning policy is:

- inject memory only for prompts still in the hard regime,
- if fresh scouts show no failures, stored negative memory should decay and usually not be injected,
- if old and new summaries disagree, the new evidence wins,
- if the memory is too old and unsupported, skip injection and retire soon,
- if the prompt is partially solved, memory may still be injected under appearance annealing until unconditioned performance catches up.

This matters because stale negative context can become harmful once the model has started to crack the prompt. The memory should guide exploration when the prompt is hard, not drag the policy back toward an old view of the prompt after the frontier has shifted.

### 4.8 Prompt Format

The memory block remains a compact descriptive wrapper prepended to the prompt.

Illustrative format:

```text
[Previous attempts on this problem repeatedly explored the following
approaches without success. Most attempts used a trigonometric substitution
after rewriting the denominator, and these paths tended to stall before
isolating a clean boxed answer. More recent attempts also tried direct
algebraic simplification, but these did not resolve the main cancellation
issue.]

{original problem prompt}
```

The output should read like a refreshed narrative, not like a log dump. The model should infer what not to keep repeating without being fed rigid prohibitions or solution hints.

### 4.9 Freshness and Staleness Management

The entire design stands or falls on freshness control.

The key principles are:

- old memory is a prior, not a verdict,
- fresh failures have priority over historical summaries,
- absence of fresh failures is evidence against continued negative conditioning,
- memory should shrink or disappear as autonomous competence emerges.

A practical heuristic stack:

1. Maintain prompt-local `solve_rate_ema`.
2. Maintain `last_updated_step` and TTL.
3. Maintain `stability_score` from how often similar failure modes reappear.
4. If `stability_score` is high and fresh failures agree with history, reinforce the memory.
5. If fresh failures differ materially from history, rewrite the memory around the new frontier.
6. If fresh scouts do not fail, decay or suppress the memory.
7. If the prompt exits the hard regime, retire the memory.

This gives the memory a simple operational meaning: it exists only while it helps characterize an active failure frontier.

### 4.10 Why This Beats Pure Stale Memory in Practice

The practical advantage over stale-only memory comes from three properties.

**Current failures re-anchor the memory to the current policy frontier.**
The model does not blindly trust a summary written many steps ago. The current scouts tell us what the policy is actually doing now.

**Old memory contributes cross-appearance depth.**
The method can accumulate evidence that a prompt tends to trigger the same losing strategy class across visits. That is exactly the information shallow RETRO lacks.

**Merge-refresh prevents frozen misconceptions from dominating.**
If the policy's failure modes evolve, the refreshed summary evolves with them. The memory is not a static instruction and not a historical transcript.

### 4.11 Why Teacher-Student Is a Later Stage

Teacher-student distillation remains a meaningful later extension:

- a memory-conditioned teacher may act as a more capable policy,
- a memory-free student may be trained to internalize that behavior,
- this may reduce inference-time dependence on memory.

But it should not be the default design for this method.

The method should first show:

1. refreshed prompt-local failure memory improves exploration directly,
2. the conditioned/unconditioned gap can be managed with dropout and annealing,
3. the gains survive at least partially when memory appearance is reduced.

Only after those are established does it make sense to add a teacher-student layer. Otherwise the distillation machinery muddies the core question.

---

## 5. Comparison of Candidate Designs

### 5.1 Current Inline RETRO Memory

Structure:

- same-step scouts,
- same-step summary,
- same-step conditioned rollout phase.

Strengths:

- freshest possible signal,
- simple causal story,
- no persistent stale state,
- strongest direct test of failure-conditioned exploration.

Weaknesses:

- limited memory depth,
- no cross-appearance accumulation,
- repeats work when the same prompt keeps reappearing with similar failures.

Assessment:

- strong baseline,
- not enough if the goal is specifically to test memory depth.

### 5.2 Persistent Post-Step Memory

Structure:

- standard GRPO step,
- summarize failures after the step,
- store summary for the next appearance.

Strengths:

- more novel than inline memory,
- simpler than an in-step refresh path,
- cheaper in immediate rollout structure.

Weaknesses:

- memory is stale at consumption time,
- the stored frontier may no longer match the current policy,
- easier to accumulate outdated negative guidance.

Assessment:

- worthwhile ablation,
- not the best practical default.

### 5.3 Teacher With Memory, Student Without Memory

Structure:

- teacher uses memory-conditioned rollouts,
- student learns to match or distill the teacher behavior.

Strengths:

- addresses memory dependence more directly,
- potentially yields a deployment policy that needs no memory at inference.

Weaknesses:

- highest complexity,
- highest compute,
- another place for instability,
- less clean attribution of what actually caused improvement.

Assessment:

- promising later stage,
- wrong first implementation target.

### 5.4 Chosen Default

The chosen default is:

> refreshed per-prompt episodic failure memory, retrieved before a prompt appearance, updated by fresh scouts, merged into a compact new record, and used to condition the second rollout phase.

This is the best practical design because it is the only one that combines:

- prompt-local persistence,
- same-step freshness,
- bounded token cost,
- explicit staleness control,
- a direct and interpretable causal path from memory to exploration behavior.

---

## 6. Evaluation Plan

### 6.1 Core Comparisons

The minimum experimental set should compare:

- baseline GRPO,
- shallow inline RETRO-GRPO,
- long refreshed-memory RETRO-GRPO.

This isolates whether memory depth improves over both the unconditioned baseline and the non-persistent failure-conditioning baseline.

### 6.2 Key Ablations

The most informative ablations are:

- refreshed-memory RETRO vs stale-only post-step memory,
- merge-refresh vs append-only memory,
- short TTL vs long TTL,
- training-policy summarizer vs frozen-base summarizer,
- memory-conditioned policy vs later-stage teacher-student distillation.

These ablations test whether the practical choices in the default design are actually necessary or merely convenient.

### 6.3 Metrics

Report not only task accuracy but also memory-specific process metrics:

- conditioned solve rate,
- unconditioned solve rate,
- conditioned/unconditioned gap,
- memory usage rate,
- memory refresh rate,
- stale-memory rate,
- prompt-token overhead,
- summary churn,
- stability score distribution.

The conditioned/unconditioned gap is especially important. If it grows while headline performance improves, the method may be creating memory dependence rather than internalized competence.

### 6.4 Benchmark Sequence

The practical evaluation sequence should remain narrow at first.

Stage 1:

- hard math only,
- deterministic verification,
- focus on hard prompts where repeated failure modes are common.

Later stages:

- agentic settings where failure frontiers may move faster,
- these are useful stress tests for stale-memory failure modes and for whether refreshed memory still helps under more dynamic transfer.

The expected prediction is:

- refreshed memory should clearly dominate stale-only memory on fast-moving benchmarks,
- on math it may show smaller but still meaningful gains through cross-appearance accumulation.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Memory becomes stale and harms search | Medium | Fresh scout refresh before use, recency weighting, TTL, retire on disagreement |
| Memory grows into a noisy archive | High if append-only | Fixed token budget, merge-refresh replacement, no raw rollout storage |
| Model becomes dependent on memory | Medium | Memory dropout, appearance annealing, monitor conditioned/unconditioned gap |
| Old negative memory suppresses now-viable strategies | Medium | New evidence wins conflicts, decay when fresh scouts no longer fail |
| Memory adds cost without gain on easy prompts | Low | Inject only in the hard regime, retire after solve-rate EMA rises |
| Teacher-student extension obscures mechanism evaluation | High if introduced early | Keep distillation as a later-stage extension rather than default design |

---

## 8. Paper Positioning

**Working title direction:** "Refreshed Failure Memory for Exploration in RL-Trained Language Models" or "Remembering What Not to Repeat: Prompt-Local Failure Memory for Hard-Problem RL"

**Core claim:** On hard prompts that recur during training, a compact prompt-local memory of prior failed strategy classes helps the model search more effectively when that memory is refreshed by current scout evidence before use. The benefit comes from combining cross-appearance persistence with same-step freshness.

**Contribution framing:**

1. **Information-state contribution:** We extend failure-conditioned exploration from same-step ephemeral summaries to refreshed prompt-local memory across appearances.
2. **Practical design contribution:** We argue that the best memory design is merge-refresh with bounded persistence, not append-only accumulation and not stale replay.
3. **Learning-dynamics contribution:** We show how memory depth can be added without abandoning the core shallow RETRO principle that guidance enters through context and learning enters through RL updates.
4. **Scope discipline:** We make a narrower and more defensible claim than general memory or continual-learning systems. The method is about search guidance on hard recurring prompts, not arbitrary long-term knowledge storage.

**Why this matters beyond the benchmark result:** The broader question is whether some exploration-relevant information should live briefly in weights, briefly in context, or in a carefully controlled combination of both. This method argues for a middle path: compact prompt-local memory as a temporary external state, refreshed by current evidence and gradually retired as the behavior internalizes.

**Related work positioning:** The method inherits the failure-conditioned exploration logic of shallow RETRO-GRPO, keeps the context-vs-gradient separation, remains distinct from positive self-conditioning and critique-based refinement, and treats teacher-student transfer as a later mechanism for internalization rather than the first-order contribution.

---

## 9. Final Practical Recommendation

The best practical design is not:

- ephemeral same-step memory alone,
- stale persistent memory alone,
- or teacher-student distillation as the first implementation.

The best practical design is:

> **refreshed per-prompt episodic failure memory**

with these defaults:

- prompt-local memory records,
- compressed descriptive summaries only,
- fresh scouts before memory use,
- merge-refresh replacement instead of append-only accumulation,
- fixed token budget,
- recency-weighted updates,
- rapid decay and retirement as prompt competence emerges,
- memory dropout and later appearance annealing to prevent dependence.

This design keeps the strongest feature of shallow RETRO-GRPO, same-step freshness, while adding the central capability that shallow RETRO lacks: memory depth across prompt appearances.
