# Swedish V2 Weight-Space Edit — Minimal Experiment

## Summary

A **pre-registered** targeted rank-1 edit to Qwen2.5-0.5B's `lm_head`,
tuned on a dev split to improve Swedish verb-second (V2) word order,
**failed**: held-out test accuracy dropped from 75% to 50% at the
dev-selected alpha (+0.25).

A **post-hoc** sweep of negative alphas on the same test set raised
accuracy to 83–92% (peak 91.7% at α=−0.5), but the dev split
*simultaneously degraded* (75% → 50–61%) at those alphas. Controls
(10 random rank-1 edits across 5 seeds, plus 2 shuffled-direction edits)
all landed at exactly baseline (75%), confirming that the *direction*
carries information. A mechanism diagnostic (verb-token overlap between
dev and test completions) traced the post-hoc improvement to a **softmax
redistribution effect over the dev/test vocabulary split**, not to a
V2-specific capability improvement.

> A rank-1 `lm_head` edit guided by Swedish activations changes held-out
> Swedish probe accuracy in a way that random or shuffled-direction edits
> do not — but the change is consistent with a vocabulary-split artifact,
> not with a genuine V2 capability improvement.

**Full investigation:** [`research_note.md`](research_note.md).ù

## Quick results

| Method | SV test | EN test | Params |
|---|---|---|---|
| Baseline | 75.0% | 100.0% | 0 |
| WSP pre-registered (α=+0.25) | 50.0% | 100.0% | rank-1 on lm_head |
| WSP post-hoc (α=−0.5) | 91.7% | 100.0% | rank-1 on lm_head |
| Random / shuffled edits | 75.0% | 100.0% | rank-1 on lm_head |
| LoRA rank-4, 200 steps | 100.0% | 83.3% | 2.2M (0.35%) |

Pre-registered WSP failed. Post-hoc WSP gain is a vocabulary-split artefact.
LoRA wins on Swedish but fails the English constraint and shows memorization.

## What this does and does not test

- **Does:** show that a rank-1 edit to `lm_head`, guided by a Swedish
  prompt direction and a dev-extracted verb list, changes held-out
  Swedish probe accuracy in a way that is *not* reproducible with 10
  random edits or 2 shuffled-direction edits.
- **Does not:** show that the edit improves Swedish V2 capability. The
  pre-registered result was a 25 pp **degradation**; the post-hoc
  improvement contradicts the dev split and is consistent with a
  vocabulary-split artifact.
- **Does not:** include a LoRA fine-tuning baseline, a fluency
  regression metric, or disjoint-vocabulary probes — all of which are
  necessary to distinguish a real V2 effect from a split artifact.

## Reproduce

```bash
pip install -r requirements.txt

# Pre-registered WSP experiment (~30 min CPU)
python code/run_experiment.py

# Post-hoc negative-α sweep
python code/additional_test.py

# Controls
python code/check1.py && python code/check2.py && python code/check3.py && python code/check4.py

# LoRA baseline (~10–20 min CPU)
python code/lora_baseline.py


