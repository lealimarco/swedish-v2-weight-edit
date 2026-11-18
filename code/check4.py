from probes import SWEDISH_V2, ENGLISH_SVO, split_probes
from steering import get_verb_token_ids
from model_utils import load_model

model, tok = load_model()
sv_dev, sv_test = split_probes(SWEDISH_V2)
dev_verbs = set(get_verb_token_ids(tok, [c for _, c, _ in sv_dev]))
test_correct = set(get_verb_token_ids(tok, [c for _, c, _ in sv_test]))
test_incorrect = set(get_verb_token_ids(tok, [c for _, _, c in sv_test]))
print("dev verbs:", sorted(dev_verbs))
print("test correct overlap with dev:", dev_verbs & test_correct)
print("test incorrect overlap with dev:", dev_verbs & test_incorrect)