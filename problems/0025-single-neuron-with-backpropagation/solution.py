import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	
	# initialization
	weights = initial_weights.copy()
	bias = initial_bias
	batch_size = len(labels)
	mse_values = []

	# training loop
	for e in range(epochs):

		# forward pass 
		logits = features @ weights + bias
		scores = 1 / (1 + np.exp(-logits))
		loss = ((labels - scores) ** 2).mean()
		mse_values.append(round(loss, 4))

		# calculate gradients
		dL_ds = -(2 / batch_size) * (labels - scores)
		ds_dlogits = scores * (1 - scores)
		dL_dlogits = dL_ds * ds_dlogits
		dL_dw = dL_dlogits @ features # (B,) x (B, N)
		dL_db = dL_dlogits.sum()

		# make update
		weights -= dL_dw * learning_rate
		bias -= dL_db * learning_rate
	
	updated_weights = (np.round(weights, 4)).tolist()
	updated_bias = round(bias, 4)

	return updated_weights, updated_bias, mse_values