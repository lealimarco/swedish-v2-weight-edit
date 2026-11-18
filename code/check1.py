from probes import SWEDISH_V2, ENGLISH_SVO, split_probes
from model_utils import load_model
from steering import make_targeted_edit
from evaluate import evaluate_on_probes, summarize

model, tok = load_model()
sv_dev, sv_test = split_probes(SWEDISH_V2)
en_dev, en_test = split_probes(ENGLISH_SVO)

for alpha in [-2.0, -1.0, -0.5, -0.25]:
    ed, _ = make_targeted_edit(model, tok, sv_dev, en_dev, alpha)
    sv = summarize(evaluate_on_probes(ed, tok, sv_dev))["accuracy_pct"]
    print(f"alpha={alpha:>5}: dev_sv={sv:.1f}%  (baseline dev ≈ 75%)")