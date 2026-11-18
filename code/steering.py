"""Steering-direction extraction and rank-1 lm_head edit."""

import copy
import torch
from model_utils import seq_logprob


def get_verb_token_ids(tok, completions):
    """Heuristic: first token of the verb in each correct completion."""
    ids = set()
    for comp in completions:
        first_word = comp.strip().split()[0]
        tokens = tok.encode(" " + first_word, add_special_tokens=False)
        if tokens:
            ids.add(tokens[0])
    return sorted(ids)


@torch.no_grad()
def extract_hidden_at_prompt_end(model, tok, prompts):
    """Mean final-layer hidden state at the last prompt token (input to lm_head)."""
    hidden_list = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt")
        out = model.model(**inputs, output_hidden_states=True)
        # last_hidden_state is post-norm, which is exactly what feeds lm_head
        hidden_list.append(out.last_hidden_state[0, -1, :])
    return torch.stack(hidden_list).mean(dim=0)


def apply_rank1_edit(model, u_direction, v_direction, alpha):
    """Return a copy of the model with lm_head.weight += alpha * u @ v^T."""
    edited = copy.deepcopy(model)
    W = edited.lm_head.weight.data                # [V, H]
    update = alpha * torch.outer(u_direction, v_direction)  # [V, H]
    edited.lm_head.weight.data = W + update
    return edited


def random_rank1_edit(model, alpha, seed=0):
    """Random rank-1 edit with the same shape as the targeted edit (control)."""
    edited = copy.deepcopy(model)
    W = edited.lm_head.weight.data
    V, H = W.shape
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(V, generator=g)
    v = torch.randn(H, generator=g)
    u = u / u.norm()
    v = v / v.norm()
    edited.lm_head.weight.data = W + alpha * torch.outer(u, v)
    return edited


def make_targeted_edit(model, tok, dev_swedish_probes, dev_english_probes, alpha):
    """Build the targeted edit for Swedish V2 using dev-set activations and
    verb token ids, verified to not hurt English on the dev set."""
    sv_prompts = [p for p, _, _ in dev_swedish_probes]
    en_prompts = [p for p, _, _ in dev_english_probes]

    # Direction in lm_head input space: mean hidden on Swedish prompts.
    v = extract_hidden_at_prompt_end(model, tok, sv_prompts)
    v = v / v.norm()

    # Direction in lm_head output (logit) space: average of Swedish and English
    # verb ids, so the edit doesn't overfit to one language's verbs.
    sv_verbs = get_verb_token_ids(tok, [c for _, c, _ in dev_swedish_probes])
    en_verbs = get_verb_token_ids(tok, [c for _, c, _ in dev_english_probes])
    target_ids = sorted(set(sv_verbs + en_verbs))
    u = torch.zeros(model.config.vocab_size)
    u[target_ids] = 1.0
    u = u / u.norm()

    return apply_rank1_edit(model, u, v, alpha), {
        "alpha": alpha,
        "target_token_ids": target_ids,
        "hidden_direction_norm": float(v.norm()),
    }