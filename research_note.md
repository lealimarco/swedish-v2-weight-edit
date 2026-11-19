# Research Note: Direction-Guided Weight Edit vs. LoRA for Swedish V2

## Hypothesis
A direction-guided rank-1 edit to `lm_head` in Qwen2.5-0.5B can raise held-out
Swedish V2 accuracy by ≥10 pp with <3 pp English SVO degradation (WSP heuristic
as a lower bound on *Sparks of Weight-Space Programming*).

## Method
- Model: `Qwen/Qwen2.5-0.5B`, fp32, CPU.
- Data: 30 synthetic Swedish V2 probes + 30 English SVO probes; 18/12 dev/test split.
- Edit: `W_lm_head += α · u ⊗ v`, where `v` = mean final-layer hidden state over
  dev Swedish prompts, `u` = verb-token indicator. α tuned on dev.
- Controls: baseline, random rank-1 (5 seeds × 2 α), shuffled-direction edit,
  rank-4 LoRA (2.2M params) on the same 18 dev probes.
- Pre-registered α grid: {+0.25, +0.5, +1.0, +2.0}.

## Results
| Condition | SV test | EN test |
|---|---|---|
| Baseline | 75.0% | 100.0% |
| WSP pre-reg (α=+0.25) | 50.0% | 100.0% |
| WSP post-hoc (α=−0.5) | 91.7% | 100.0% |
| Random / shuffled edits | 75.0% | 100.0% |
| LoRA rank-4, 200 steps | 100.0% | 83.3% |

Pre-registered WSP failed (Swedish −25 pp). Post-hoc negative-α peak (+16.7 pp)
disappears under a verb-overlap diagnostic: dev/test verb vocabularies overlap
5/12, and softmax redistribution accounts for the +2 items without a V2 mechanism.
LoRA on the same 18 examples reaches 100% but trades 16.7 pp English and shows
loss collapse (3e-4), consistent with template memorization.

## Interpretation
1. Edit **direction** matters: targeted edits move test by 15–17 pp; 10 random
   and 2 shuffled edits sit exactly at baseline.
2. The WSP heuristic is **not competitive** with a data-matched LoRA baseline.
3. The post-hoc WSP gain is a **split artefact**, not a V2 improvement.
4. The source paper's obstacle-vs-absence dichotomy needs a third category:
   *capability present but heuristic too weak to exploit it*.

## Recommendation
Do not pursue WSP as a Swedish V2 fix on this evidence. The single most
informative next experiment is a **disjoint-vocabulary probe set (200 items/lang)**
run against both WSP and LoRA. That one experiment distinguishes "both methods
exploit template structure" from "LoRA generalizes, WSP does not" from
"WSP does something LoRA does not" — three outcomes, each actionable.

## Limitations
n=12 test items; LoRA is data-matched and memorizes; no fluency (perplexity)
regression; no disjoint-vocabulary probes. The negative result is about the
**heuristic**, not about the full LLM-over-weights pipeline in the source paper.