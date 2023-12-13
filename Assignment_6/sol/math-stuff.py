import numpy as np


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


W_1 = np.array([[-0.2, 0.9, 0.4], [-0.1, 0.3, 0.4], [0.2, 0.5, -0.7], [0.2, -0.5, 0.5]])

W_2 = np.array([[0.6, -0.2], [-0.1, 0.8], [-0.5, -0.3]])

x = np.array([3, 1, -1, 2])

h = W_1.T @ x

print(f"h = {h}")

o = W_2.T @ h

print(f"o = {o}")

print(f"softmax(o) = {softmax(o)}")
