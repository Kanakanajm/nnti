import numpy as np


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

import numpy as np

def cross_entropy_loss_function(predictions, true_labels, epsilon=1e-12):
    # To avoid log(0), we add a small epsilon value to predictions
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    # Calculate log of probabilities
    log_predictions = np.log(predictions)
    
    # Assume true_labels are one-hot encoded, we use np.sum to compute the dot product between the log of the predictions and the true labels
    loss = -np.sum(true_labels * log_predictions, axis=1)
    
    return loss

W_hidden = np.array([[0.15, -0.25, 0.05], [0.2, 0.1, -0.15]])

W_out = np.array([[0.2, 0.5], [-0.35, 0.15], [0.15, -0.2]])

x = np.array([-1, 1])

true_o = np.array([[1, 0]])

# 1st Forward-Pass

z_1 = W_hidden.T @ x

print(f"z_1 = {z_1}")

a_1 = h = leaky_relu(z_1)

print(f"a_1 = h = leaky_relu(z_1) = {h}")

z_2 = W_out.T @ a_1

print(f"z_2 = {z_2}")

a_2 = o = softmax(z_2)

print(f"a_2 = o = softmax(z_2) = {softmax(z_2)}")

print(f"L(1) = {cross_entropy_loss_function(o.reshape(1, -1), true_o)}")
