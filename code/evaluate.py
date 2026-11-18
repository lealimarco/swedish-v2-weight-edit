"""Evaluation helpers."""

import pandas as pd
from model_utils import v2_preference, seq_logprob


def evaluate_on_probes(model, tok, probes):
    """Return list of dicts with per-item correctness."""
    rows = []
    for prompt, correct, incorrect in probes:
        pref = v2_preference(model, tok, prompt, correct, incorrect)
        rows.append({
            "prompt": prompt,
            "correct_completion": correct,
            "incorrect_completion": incorrect,
            "prefers_correct": pref,
            "lp_correct": seq_logprob(model, tok, prompt, correct),
            "lp_incorrect": seq_logprob(model, tok, prompt, incorrect),
        })
    return rows


def summarize(rows):
    if not rows:
        return {"n": 0, "accuracy_pct": 0.0}
    correct = sum(r["prefers_correct"] for r in rows)
    return {"n": len(rows), "accuracy_pct": 100.0 * correct / len(rows)}


def to_dataframe(rows, condition, language):
    df = pd.DataFrame(rows)
    df["condition"] = condition
    df["language"] = language
    return df