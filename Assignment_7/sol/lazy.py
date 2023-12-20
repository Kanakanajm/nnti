import numpy as np

x = np.array([[-1], [1]])
o = np.array([[1], [0]])

wh = np.array([[0.15, -0.25, 0.05],
               [0.2, 0.1, -0.15]])

wo = np.array([[0.2, 0.5],
               [-0.35, 0.15],
               [0.15, -0.2]])

def relu(x):
    return np.maximum(x, 0.01*x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def mult(w, a):
    return w.T @ a

def forward(x):
    return softmax(mult(wo, relu(mult(wh, x))))



print(forward(x))