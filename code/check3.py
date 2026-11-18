
import torch
from probes import SWEDISH_V2, ENGLISH_SVO, split_probes
from model_utils import load_model
from steering import extract_hidden_at_prompt_end, apply_rank1_edit, get_verb_token_ids
from evaluate import evaluate_on_probes, summarize

model, tok = load_model()
sv_dev, sv_test = split_probes(SWEDISH_V2)
en_dev, en_test = split_probes(ENGLISH_SVO)

v_en = extract_hidden_at_prompt_end(model, tok, [p for p, _, _ in en_dev])
v_en = v_en / v_en.norm()

sv_verbs = get_verb_token_ids(tok, [c for _, c, _ in sv_dev])
u = torch.zeros(model.config.vocab_size)
u[sv_verbs] = 1.0
u = u / u.norm()

for alpha in [-0.5, -1.0]:
    ed = apply_rank1_edit(model, u, v_en, alpha)
    sv = summarize(evaluate_on_probes(ed, tok, sv_test))["accuracy_pct"]
    print(f"shuffled EN-direction  alpha={alpha:>5}  test_sv={sv:.1f}%")