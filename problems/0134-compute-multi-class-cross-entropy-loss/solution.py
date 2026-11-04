import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    clipped = np.clip(predicted_probs, epsilon, 1-epsilon)
    result = np.log(clipped) * true_labels
    return -(np.sum(result, axis=1)).mean()