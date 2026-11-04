import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    numerator = torch.tensor(scores)
    numerator -= torch.max(numerator)
    numerator = torch.exp(numerator)
    denominator = torch.sum(numerator)
    return (numerator / denominator).tolist()
