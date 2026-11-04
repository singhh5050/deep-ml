import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	logits = np.array(scores, dtype=float)
	norm_logits = logits - np.max(logits)
	log_probs = norm_logits - np.log(np.sum(np.exp(norm_logits)))
	return log_probs