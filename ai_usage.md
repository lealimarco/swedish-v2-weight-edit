
---

## `ai_usage.md`

```markdown
# AI Usage

## Tools used

- **Frontier LLM (Claude, Anthropic)** — used for:
  - Critical analysis of the source blog post "Sparks of Weight-Space
    Programming".
  - Hypothesis framing and experimental design (pre-registered alpha
    grid, controls, evaluation protocol).
  - Initial code skeleton for `probes.py`, `model_utils.py`,
    `steering.py`, `evaluate.py`, `run_experiment.py`, and the four
    follow-up check scripts.
  - Drafting `research_note.md` and `README.md`.

- **No model API calls** are made by the experiment itself. The
  "targeted edit" is a hand-crafted heuristic (mean hidden state + verb
  token IDs), not a frontier-LLM-generated edit.

- **No external data** was used. Probes are synthetic (30 Swedish V2 +
  30 English SVO items), split 60/40 into dev/test.

## What the AI did not do

- Run any experiments. All reported results are from runs on the
  author's machine (CPU, Qwen2.5-0.5B, ~30 minutes total wall time).
- Select the final narrative. The decision to report the pre-registered
  failure first and the post-hoc exploration second is the author's,
  made after seeing the results and following the assignment's stated
  preference for honest negative results.
- Write the mechanism explanation unaided. The softmax-redistribution
  story emerged from the `check4.py` verb-overlap diagnostic that was
  run by the author and then interpreted jointly with the assistant.

## Independence note

The pre-registered experiment (`run_experiment.py`) and its output
(`results/summary.json`) were produced from the code exactly as
generated, without post-hoc tuning. All negative-alpha experiments are
labelled as post-hoc and exploratory in `research_note.md` Section 4.2.

## Reproducibility

All code is committed as-is. Model revision is the default
`Qwen/Qwen2.5-0.5B` from Hugging Face Hub at the time of the experiment.
Random seed for `random_rank1_edit` is fixed per call; seeds are passed
explicitly in `check2.py` (0–4). Determinism of the targeted edit is
guaranteed by `torch.manual_seed(42)` in `load_model`.