import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    weights = weights.copy()
    if method == "stochastic":
        for e in range(n_epochs):
            for i in range(len(y)):
                logit = X[i] @ weights
                loss = (y[i] - logit) ** 2
                
                dloss_dlogit = 2 * (y[i] - logit) * (-1) 
                dlogit_dW = X[i]
                dloss_dW = dloss_dlogit * dlogit_dW

                weights -= dloss_dW * learning_rate
    elif method == "batch":
        for e in range(n_epochs):
            logits = X @ weights # (B x N) (N x 1) => (B)
            loss = ((y - logits) ** 2).mean()

            dloss_dlogits = (1 / len(y)) * 2 * (y - logits) * (-1)
            dlogits_dW = X
            dloss_dW = dloss_dlogits @ dlogits_dW # (1, B) x (B x N)
            weights -= dloss_dW * learning_rate
    elif method == "mini_batch":
        for e in range(n_epochs):
            for i in range(0, len(y), batch_size):
                logits = X[i:i+batch_size] @ weights
                loss = ((y[i:i+batch_size] - logits) ** 2).mean()

                dloss_dlogits = (1 / batch_size) * 2 * (y[i:i+batch_size] - logits) * (-1)
                dlogits_dW = X[i:i+batch_size]
                dloss_dW = dloss_dlogits @ dlogits_dW
                weights -= dloss_dW * learning_rate
    return weights
