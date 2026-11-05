import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
	average = X.mean(dim=-1, keepdim=True)
	variance = ((X - average)**2).mean(dim=-1,  keepdim=True)
	normalized = (X - average) / (variance + epsilon).sqrt()
	return gamma * normalized + beta