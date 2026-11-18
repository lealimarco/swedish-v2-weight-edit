"""Model loading and sequence log-probability computation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name="Qwen/Qwen2.5-0.5B", seed=42, untie_lm_head=True):
    """Load model and tokenizer. Optionally untie lm_head so edits don't
    propagate into input embeddings (Qwen2.5-0.5B ties them by default)."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    if untie_lm_head:
        model.lm_head.weight = torch.nn.Parameter(
            model.lm_head.weight.data.clone()
        )
    model.eval()
    return model, tok


@torch.no_grad()
def seq_logprob(model, tok, prompt: str, completion: str) -> float:
    """Return log P(completion | prompt) as a scalar (CPU tensor -> float)."""
    full = prompt + completion
    inputs = tok(full, return_tensors="pt")
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]

    logits = model(**inputs).logits          # [1, T, V]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = inputs.input_ids[:, 1:]     # [1, T-1]
    token_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)[0]

    # Completion tokens start at index n_prompt-1 in the shifted sequence.
    return token_lp[n_prompt - 1:].sum().item()


@torch.no_grad()
def v2_preference(model, tok, prompt, correct, incorrect) -> int:
    """1 if model prefers the correct (V2/SVO) completion, else 0."""
    lp_correct = seq_logprob(model, tok, prompt, correct)
    lp_incorrect = seq_logprob(model, tok, prompt, incorrect)
    return int(lp_correct > lp_incorrect)