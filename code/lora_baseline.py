"""
Rank-4 LoRA baseline on the same 18 Swedish dev probes used to build
the targeted rank-1 edit. Evaluates on the same held-out test splits.

Run from the code/ directory:
    python3 lora_baseline.py

Expected runtime on CPU for Qwen2.5-0.5B: ~10-20 minutes.
"""

import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator
from peft import LoraConfig, get_peft_model, TaskType

from probes import SWEDISH_V2, ENGLISH_SVO, split_probes
from model_utils import load_model, seq_logprob, v2_preference
from evaluate import evaluate_on_probes, summarize

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 4
LORA_ALPHA = 8          # 2 * rank, standard heuristic
LORA_DROPOUT = 0.05
LR = 1e-3
STEPS = 200
BATCH_SIZE = 4
SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class ProbeDataset(Dataset):
    """Each item: prompt + correct completion. Loss only on completion tokens."""

    def __init__(self, probes, tok, max_length=128):
        self.items = []
        for prompt, correct, _ in probes:
            enc = tok(prompt + correct, return_tensors="pt",
                       truncation=True, max_length=max_length)
            input_ids = enc.input_ids[0]
            n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
            labels = input_ids.clone()
            labels[:n_prompt] = -100          # mask prompt tokens
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train_lora(model, tok, probes, steps=STEPS, lr=LR):
    """Attach rank-4 LoRA and train on prompt+correct completions."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    ds = ProbeDataset(probes, tok)
    # Pad to max length in the batch
    def collate(batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids, labels = [], []
        for x in batch:
            pad = max_len - x["input_ids"].shape[0]
            input_ids.append(torch.nn.functional.pad(
                x["input_ids"], (0, pad), value=tok.pad_token_id or 0))
            labels.append(torch.nn.functional.pad(
                x["labels"], (0, pad), value=-100))
        return {"input_ids": torch.stack(input_ids),
                "attention_mask": (torch.stack(input_ids) !=
                                   (tok.pad_token_id or 0)).long(),
                "labels": torch.stack(labels)}

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate, drop_last=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    step = 0
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            optim.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optim.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps}  loss={out.loss.item():.4f}")
    model.eval()
    return model


def main():
    torch.manual_seed(SEED)
    print(f"Loading {MODEL_NAME} ...")
    model, tok = load_model(MODEL_NAME)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sv_dev, sv_test = split_probes(SWEDISH_V2)
    en_dev, en_test = split_probes(ENGLISH_SVO)
    print(f"Swedish: {len(sv_dev)} dev / {len(sv_test)} test")
    print(f"English: {len(en_dev)} dev / {len(en_test)} test")

    # Baseline for reference
    print("\n[Baseline]")
    base_sv = summarize(evaluate_on_probes(model, tok, sv_test))
    base_en = summarize(evaluate_on_probes(model, tok, en_test))
    print(f"  Swedish V2: {base_sv['accuracy_pct']:.1f}%")
    print(f"  English SVO: {base_en['accuracy_pct']:.1f}%")

    # Train LoRA on the same Swedish dev probes
    print(f"\n[LoRA rank-{LORA_RANK} training on {len(sv_dev)} dev probes]")
    lora_model = train_lora(model, tok, sv_dev)

    # Evaluate on held-out test sets
    print("\n[LoRA evaluation on held-out test]")
    lora_sv = summarize(evaluate_on_probes(lora_model, tok, sv_test))
    lora_en = summarize(evaluate_on_probes(lora_model, tok, en_test))
    print(f"  Swedish V2: {lora_sv['accuracy_pct']:.1f}%")
    print(f"  English SVO: {lora_en['accuracy_pct']:.1f}%")

    n_trainable = sum(p.numel() for p in lora_model.parameters()
                     if p.requires_grad)

    summary = {
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "steps": STEPS,
        "lr": LR,
        "trainable_params": n_trainable,
        "baseline_swedish": base_sv["accuracy_pct"],
        "baseline_english": base_en["accuracy_pct"],
        "lora_swedish": lora_sv["accuracy_pct"],
        "lora_english": lora_en["accuracy_pct"],
    }
    out_path = os.path.join(RESULTS_DIR, "lora_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()