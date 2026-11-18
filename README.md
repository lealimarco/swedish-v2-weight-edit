# Swedish V2 Weight-Space Edit — Minimal Experiment

Minimal CPU-only experiment testing whether a **targeted rank-1 edit to
Qwen2.5-0.5B's `lm_head`**, guided by model activations on Swedish
verb-second (V2) prompts, improves Swedish V2 word-order preference on
held-out probes **without degrading English SVO** performance.

## Hypothesis

The Swedish V2 failure in a small multilingual model is **localisable**: a
small, direction-guided weight edit removes the English-defaulting failure
and lets the existing V2 capability express itself, without hurting English.

## What this does and does not test

- **Does:** show that a rank-1 edit to `lm_head`, guided by a Swedish prompt
  direction and a dev-extracted verb list, changes held-out Swedish probe
  accuracy (75% → 91.7% at α=−0.5), and that this change is not reproducible
  with 10 random edits or 2 shuffled-direction edits (all flat at 75%).
- **Does not:** show that the edit improves Swedish V2 capability. Dev
  accuracy degrades at the same α values; the target verb list overlaps
  with *correct* test answers, not incorrect ones; the +2 test items are
  consistent with a softmax redistribution effect specific to the
  dev/test vocabulary split.
- **Does not:** include a LoRA baseline, a fluency regression metric, or
  disjoint-vocabulary probes — all of which are necessary to distinguish
  a real V2 effect from a split artefact.

## Reproduce

```bash
pip install -r requirements.txt
cd code
python run_experiment.py
```


