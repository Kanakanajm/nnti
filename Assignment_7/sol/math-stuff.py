import numpy as np


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def cross_entropy_loss_function(predictions, true_labels, epsilon=1e-12):
    # To avoid log(0), we add a small epsilon value to predictions
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    # Calculate log of probabilities
    log_predictions = np.log(predictions)

    # Assume true_labels are one-hot encoded, we use np.sum to compute the dot product between the log of the predictions and the true labels
    loss = -np.sum(true_labels * log_predictions, axis=1)

    return loss


def mult(w, a):
    return w.T @ a


# Parameters
W_hidden = w1 = np.array([[0.15, -0.25, 0.05], [0.2, 0.1, -0.15]])
W_out = w2 = np.array([[0.2, 0.5], [-0.35, 0.15], [0.15, -0.2]])
x = np.array([-1, 1])
y_true = np.array([[1, 0]])
alpha = 0.1
max_iter = 1000
for i in range(max_iter):
    # Forward-Pass
    z1 = mult(w1, x)
    a1 = leaky_relu(z1)
    z2 = mult(w2, a1)
    a2 = softmax(z2)
    print(
        "iteration",
        i,
        "\tprediction",
        a2.tolist(),
        "\tloss",
        cross_entropy_loss_function(a2, y_true),
    )

    # Backward-Pass
    dL_dz2 = a2 - y_true
    dz2_dw2 = a1.copy().reshape(-1, 1)
    dL_dw2 = dz2_dw2 @ dL_dz2

    dz2_da1 = w2.copy()
    dL_da1 = dz2_da1 @ dL_dz2.T

    da1_dz1 = z1.copy().reshape(-1, 1)
    da1_dz1[da1_dz1 > 0] = 1
    da1_dz1[da1_dz1 <= 0] = 0.01

    dz1_dw1 = x.reshape(-1, 1)

    dL_dw1 = dz1_dw1 @ (dL_da1 * da1_dz1).T

    w1 -= alpha * dL_dw1
    w2 -= alpha * dL_dw2
