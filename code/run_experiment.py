"""Main experiment driver.

Tests whether a targeted rank-1 edit to Qwen2.5-0.5B's lm_head improves
Swedish V2 word-order preference on held-out probes without degrading
English SVO performance.
"""

import json
import os

import pandas as pd

from probes import SWEDISH_V2, ENGLISH_SVO, split_probes
from model_utils import load_model
from steering import make_targeted_edit, random_rank1_edit
from evaluate import evaluate_on_probes, summarize, to_dataframe


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ALPHA = 1.0           # rank-1 edit strength; tuned on dev set below
ALPHA_GRID = [0.25, 0.5, 1.0, 2.0]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tok = load_model(MODEL_NAME)

    sv_dev, sv_test = split_probes(SWEDISH_V2)
    en_dev, en_test = split_probes(ENGLISH_SVO)
    print(f"Swedish: {len(sv_dev)} dev / {len(sv_test)} test")
    print(f"English: {len(en_dev)} dev / {len(en_test)} test")

    # ----- Baseline -----
    print("\n[Baseline]")
    baseline_sv = evaluate_on_probes(model, tok, sv_test)
    baseline_en = evaluate_on_probes(model, tok, en_test)
    baseline_sv_sum = summarize(baseline_sv)
    baseline_en_sum = summarize(baseline_en)
    print(f"  Swedish V2 acc: {baseline_sv_sum['accuracy_pct']:.1f}%")
    print(f"  English SVO acc: {baseline_en_sum['accuracy_pct']:.1f}%")

    # ----- Tune alpha on dev set -----
    print("\n[Tuning alpha on dev set]")
    best_alpha = None
    best_dev_gain = -1e9
    for alpha in ALPHA_GRID:
        edited, _ = make_targeted_edit(model, tok, sv_dev, en_dev, alpha)
        sv_dev_acc = summarize(evaluate_on_probes(edited, tok, sv_dev))["accuracy_pct"]
        en_dev_acc = summarize(evaluate_on_probes(edited, tok, en_dev))["accuracy_pct"]
        print(f"  alpha={alpha}: sv_dev={sv_dev_acc:.1f}%  en_dev={en_dev_acc:.1f}%")
        # Require no more than 3pp English loss on dev
        if en_dev_acc >= summarize(evaluate_on_probes(model, tok, en_dev))["accuracy_pct"] - 3.0:
            if sv_dev_acc - baseline_sv_sum["accuracy_pct"] > best_dev_gain:
                best_dev_gain = sv_dev_acc - baseline_sv_sum["accuracy_pct"]
                best_alpha = alpha

    if best_alpha is None:
        best_alpha = 0.5
        print(f"  No alpha preserved English on dev; falling back to {best_alpha}")
    else:
        print(f"  Best alpha on dev: {best_alpha}")

    # ----- Targeted edit at best alpha -----
    print(f"\n[Targeted edit] alpha={best_alpha}")
    targeted_model, meta = make_targeted_edit(model, tok, sv_dev, en_dev, best_alpha)
    targeted_sv = evaluate_on_probes(targeted_model, tok, sv_test)
    targeted_en = evaluate_on_probes(targeted_model, tok, en_test)
    targeted_sv_sum = summarize(targeted_sv)
    targeted_en_sum = summarize(targeted_en)
    print(f"  Swedish V2 acc: {targeted_sv_sum['accuracy_pct']:.1f}%")
    print(f"  English SVO acc: {targeted_en_sum['accuracy_pct']:.1f}%")

    # ----- Random edit control -----
    print(f"\n[Random rank-1 edit] alpha={best_alpha}")
    random_model = random_rank1_edit(model, best_alpha, seed=1)
    random_sv = evaluate_on_probes(random_model, tok, sv_test)
    random_en = evaluate_on_probes(random_model, tok, en_test)
    random_sv_sum = summarize(random_sv)
    random_en_sum = summarize(random_en)
    print(f"  Swedish V2 acc: {random_sv_sum['accuracy_pct']:.1f}%")
    print(f"  English SVO acc: {random_en_sum['accuracy_pct']:.1f}%")

    # ----- Save results -----
    summary = {
        "model": MODEL_NAME,
        "best_alpha": best_alpha,
        "conditions": {
            "baseline": {
                "swedish_v2_acc": baseline_sv_sum["accuracy_pct"],
                "english_svo_acc": baseline_en_sum["accuracy_pct"],
            },
            "targeted_edit": {
                "swedish_v2_acc": targeted_sv_sum["accuracy_pct"],
                "english_svo_acc": targeted_en_sum["accuracy_pct"],
            },
            "random_edit": {
                "swedish_v2_acc": random_sv_sum["accuracy_pct"],
                "english_svo_acc": random_en_sum["accuracy_pct"],
            },
        },
        "edit_metadata": meta,
    }
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    frames = []
    for cond, sv_rows, en_rows in [
        ("baseline", baseline_sv, baseline_en),
        ("targeted_edit", targeted_sv, targeted_en),
        ("random_edit", random_sv, random_en),
    ]:
        frames.append(to_dataframe(sv_rows, cond, "swedish"))
        frames.append(to_dataframe(en_rows, cond, "english"))
    pd.concat(frames).to_csv(f"{RESULTS_DIR}/per_item_results.csv", index=False)

    print(f"\nResults written to {RESULTS_DIR}/summary.json and per_item_results.csv")
    print("\n=== FINAL SUMMARY ===")
    print(pd.DataFrame([
        {"condition": "baseline",
         "sv_acc": baseline_sv_sum["accuracy_pct"],
         "en_acc": baseline_en_sum["accuracy_pct"]},
        {"condition": "targeted_edit",
         "sv_acc": targeted_sv_sum["accuracy_pct"],
         "en_acc": targeted_en_sum["accuracy_pct"]},
        {"condition": "random_edit",
         "sv_acc": random_sv_sum["accuracy_pct"],
         "en_acc": random_en_sum["accuracy_pct"]},
    ]))


if __name__ == "__main__":
    main()