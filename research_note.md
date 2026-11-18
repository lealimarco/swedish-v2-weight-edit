# Can a direction-guided weight edit fix Swedish V2 in a 0.5B model?

## 1. Critical examination of the research

The blog post's strongest evidence is the Qwen3.5-0.8B reasoning-loop edit
(0.016% of weights → 7.9 → 48.1% on MATH-500). Its most important
limitation is that every successful edit was produced by a frontier LLM
(GPT-6 Astra) whose reasoning traces are not reproducible without access
to that model. The strongest negative result — Gemma 270M, where no edit
produced English-in-Italian — is arguably more informative for SE.LLMA,
because it shows the approach only works when the target capability is
**already present and merely blocked**.

For a Swedish model, the question is not whether a frontier LLM can write
weight surgery for a reasoning loop, but whether a small **targeted edit**
can remove a **Swedish-specific** failure without degrading other
languages. Swedish verb-second (V2) word order is a good candidate: after
a fronted adverbial, the finite verb must come second, and small
multilingual models often default to English SVO order
(*"Igår jag gick"* instead of *"Igår gick jag"*).

## 2. Hypothesis

H1: The Swedish V2 failure in Qwen2.5-0.5B is localisable — a small,
direction-guided rank-1 edit to `lm_head` raises Swedish V2 accuracy on
held-out probes by ≥ 10 pp with < 3 pp English degradation.

**Supportive evidence:** targeted edit ≥ baseline + 10 pp on Swedish test,
English test drops < 3 pp, and the random edit does not show the same gain.

**Challenging evidence:** random edit produces an equal or larger Swedish
gain; English degrades more than 3 pp; alpha tuning cannot find a setting
that improves Swedish on the dev split.

## 3. Experiment

- Model: `Qwen/Qwen2.5-0.5B`, CPU, `lm_head` untied from input embeddings.
- Data: 30 synthetic Swedish V2 probes + 30 English SVO probes, each split
  60/40 into dev (18/18) and test (12/12).
- Edit: rank-1 update `W_lm_head += alpha · u ⊗ v`, where `v` is the mean
  final-layer hidden state on dev Swedish prompts and `u` is a unit vector
  over verb token IDs extracted from dev completions.
- Pre-registered alpha grid: `{+0.25, +0.5, +1.0, +2.0}`, chosen on the
  dev split subject to English dev loss < 3 pp.
- Controls: unchanged baseline; random rank-1 edit at the same `alpha`.
- Metrics: probe accuracy (V2 vs SVO preference by log-probability).

### 3.1 Investigation timeline

The investigation happened in two phases. Reporting the timeline
explicitly is important for interpreting the results honestly.

1. **Pre-registered main experiment** (Section 4.1): positive alpha grid
   {+0.25, +0.5, +1.0, +2.0}, dev-tuned, evaluated on held-out test.
   → Hypothesis failed. Targeted edit dropped test accuracy by 25 pp.
2. **Dev diagnostic at negative α** (Section 4.3): confirmed dev also
   degrades (75% → 61% at α=−0.5, 50% at α=−1.0).
3. **Exploratory test sweep at negative α** (Section 4.2): test improved
   (75% → 83–92%). This contradicted step 2 and motivated the controls.
4. **Controls** (Section 4.4): random rank-1 edits (10 runs, 5 seeds × 2
   alphas) and shuffled-direction edits (2 runs) all left test at exactly
   75.0%. The direction matters.
5. **Mechanism diagnostic** (Section 4.5): verb-overlap check showed dev
   verbs overlap with *correct* test verbs (5/12) and not with incorrect
   ones (0/12). The test improvement is consistent with softmax
   redistribution over the dev/test vocabulary split, not with a V2
   capability improvement.

## 4. Results

### 4.1 Pre-registered main experiment (positive alpha grid, dev-tuned)

Dev accuracy at the pre-registered grid (18 items):

| α    | dev Swedish | dev English |
|------|-------------|-------------|
| +0.25 | 44.4%      | 100.0%      |
| +0.50 | 44.4%      | 94.4%       |
| +1.00 | 44.4%      | 94.4%       |
| +2.00 | 38.9%      | 83.3%       |

Best dev α = **+0.25** (highest Swedish, English preserved). On held-out
test (12 items):

| Condition         | Swedish V2 | English SVO |
|-------------------|-----------|-------------|
| baseline          | 75.0%     | 100.0%      |
| targeted @ α=+0.25| **50.0%** | 100.0%      |
| random @ α=+0.25  | 75.0%     | 100.0%      |

**The main hypothesis failed.** The targeted edit decreased test accuracy
by 25 pp. The random control was flat, confirming the drop was due to the
*direction*, not the magnitude of the perturbation. This is the
pre-registered result, and it is what `run_experiment.py` and
`results/summary.json` report.

### 4.2 Post-hoc exploratory sweep (negative alphas, test set)

After the failure above, an exploratory sweep of negative alphas on the
same held-out test set gave:

| α     | Swedish V2 | English SVO |
|-------|-----------|-------------|
| +0.10 | 75.0%     | 100.0%      |
| −0.25 | 83.3%     | 100.0%      |
| −0.50 | **91.7%** | 100.0%      |
| −1.00 | 83.3%     | 100.0%      |
| −2.00 | 83.3%     | 100.0%      |

The test improvement at negative α is **exploratory and post-hoc**. It was
observed after the pre-registered experiment failed, on the same test set,
and it prompted the diagnostic investigation in Sections 4.3–4.5.

### 4.3 Dev/test disagreement

Dev accuracy at the same negative alphas (in-sample for the edit):

| α     | dev Swedish |
|-------|-------------|
| −0.25 | 61.1%       |
| −0.50 | 61.1%       |
| −1.00 | 50.0%       |
| −2.00 | 50.0%       |

The targeted edit *improves* test accuracy (75% → 83–92%) but *degrades*
dev accuracy (75% → 50–61%) at the same alphas. With dev = 18 items and
test = 12 items, a single item flip is 5.6 pp on dev and 8.3 pp on test.
The disagreement cannot be resolved at this sample size by accuracy alone.

### 4.4 Controls

| Control                                                          | Swedish V2 (test) |
|------------------------------------------------------------------|-------------------|
| Baseline                                                         | 75.0%             |
| Random rank-1 edit, α=−0.5 and −1.0, 5 seeds (10 runs)           | 75.0% (every run) |
| Shuffled direction: English activations × Swedish verb tokens, α=−0.5 and −1.0 | 75.0% (both)      |

10 random edits and 2 shuffled-direction edits all land at exactly
baseline. The direction is doing work; magnitude alone is not.

### 4.5 Mechanism diagnostic

`get_verb_token_ids` collects the first token of each finite verb in dev
correct completions. Overlap with test completions:

- Dev verbs ∩ test *correct* verbs: **5 tokens** (5/12 test items).
- Dev verbs ∩ test *incorrect* verbs: **0 tokens** (incorrect SVO
  completions begin with pronouns, not verbs).

Under a naive "negative α suppresses the SVO answer" model, test accuracy
should stay flat or drop. It rises by 2 items (75% → 91.7%). The plausible
explanation is **softmax redistribution**: decreasing dev-verb logits
raises the *relative* probability of every other token, including the
correct verbs of the 7/12 test items whose verbs are not in the dev
list. If 2 of those 7 cross the preference threshold, we get exactly the
observed +2 items.

## 5. Interpretation

The pre-registered experiment failed and the post-hoc improvement is best
explained as a **vocabulary-split artifact**, not a V2 capability
improvement.

- **Dev is constructed to fail under negative α.** Every dev correct
  answer uses a dev verb, so the edit attacks every dev correct answer.
  This is why dev accuracy drops from 75% to 50–61% at every negative α.
- **Test partially benefits** because 7/12 test correct answers use verbs
  outside the dev list and are relatively lifted by the suppression of
  the 5 dev-overlapping verbs.
- **The controls still support a narrow methodological claim:**
  random edits and shuffled-direction edits leave test accuracy at
  exactly baseline, so the *direction* carries information — it is not
  noise and not a pure magnitude effect.
- **But what the direction does is not "improve Swedish V2."** It
  redistributes probability mass over the dev/test vocabulary split. The
  dev split disagrees with test in sign, and the mechanism diagnostic
  points to vocabulary coverage rather than V2 syntax.

This result mirrors the paper's Gemma 270M negative finding more than its
Qwen3.5 positive finding. WSP can remove an *obstacle* when the target
capability is already present and cleanly blocked. Here, V2 at baseline is
75% correct; there is no clean obstacle to remove, and any edit that
perturbs the `lm_head` at this scale mostly reshuffles which verbs win,
not whether V2 wins.

The honest claim this experiment supports:

> A rank-1 `lm_head` edit guided by Swedish prompt activations and a
> dev-extracted verb-token list changes held-out Swedish probe accuracy
> in a way that is not reproducible by random or shuffled-direction
> edits. The change is consistent with a softmax redistribution effect
> over the dev/test vocabulary split, not with a V2-specific capability
> improvement.

## 6. Recommendation

**Do not treat this as evidence that WSP fixes Swedish V2.** The
pre-registered experiment failed, and the post-hoc improvement is
explained by a data-split artifact rather than a V2 capability change.

The one durable positive finding is methodological: **direction matters**.
A rank-1 edit guided by Swedish activations moves behaviour; identical-
magnitude random edits and a direction derived from English activations
do not. That is real signal about WSP as a *mechanism*, not about WSP as
a *fix*.

**Most informative next experiment — the one that would actually test
whether WSP fixes V2:**

1. **Disjoint-vocabulary probes.** Build 200 V2 items per language where
   the dev and test splits sample verbs from disjoint vocabularies
   (dev verbs A, test verbs B). If the effect disappears when the split
   is disjoint, it confirms the vocabulary-split explanation and the
   post-hoc claim collapses to a data artifact. If it persists, WSP is
   doing something to V2 itself.
2. **Difference-in-means direction.** Replace the mean-direction `v`
   with the difference-in-means between V2-correct and V2-incorrect
   activations on dev. This controls for the mean-direction's inability
   to distinguish "this is a Swedish prompt" from "this prompt should
   trigger V2."
3. **LoRA fine-tuning baseline** on the same probe sets. If LoRA at 10
   minutes gets 90%+ on disjoint-vocabulary test items while WSP does
   not, WSP is not a competitive alternative for this task.
4. **Fluency regression check** (perplexity on 500 English sentences).
   English probe accuracy at 100% can hide a fluency drop.

Until (1) and (2) are done, this submission should be read as an honest
**negative result about the naively-configured method**, with a positive
methodological side-finding about the direction's specificity.