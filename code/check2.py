
from probes import SWEDISH_V2, ENGLISH_SVO, split_probes
from model_utils import load_model
from steering import random_rank1_edit
from evaluate import evaluate_on_probes, summarize

model, tok = load_model()
sv_dev, sv_test = split_probes(SWEDISH_V2)
en_dev, en_test = split_probes(ENGLISH_SVO)

base_sv = summarize(evaluate_on_probes(model, tok, sv_test))["accuracy_pct"]
print(f"baseline test_sv = {base_sv:.1f}%")

for seed in [0, 1, 2, 3, 4]:
    for alpha in [-0.5, -1.0]:
        r = random_rank1_edit(model, alpha, seed=seed)
        sv = summarize(evaluate_on_probes(r, tok, sv_test))["accuracy_pct"]
        print(f"seed={seed}  alpha={alpha:>5}  random_test_sv={sv:.1f}%")