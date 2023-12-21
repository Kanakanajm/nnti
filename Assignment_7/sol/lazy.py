import numpy as np

def relu(x):
    return np.maximum(x, 0.01*x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def mult(w, a):
    return w.T @ a

def loss(a, y):
    return -np.sum(y*np.log(a))

alpha = 0.1

x = np.array([[-1], [1]])
y = np.array([[1], [0]])

w1 = np.array([[0.15, -0.25, 0.05],
               [0.2, 0.1, -0.15]])

w2 = np.array([[0.2, 0.5],
               [-0.35, 0.15],
               [0.15, -0.2]])

for i in range(10):
    z1 = mult(w1, x)
    a1 = relu(z1)
    z2 = mult(w2, a1)
    a2 = softmax(z2)
    print("iteration", i, "\tprediction", a2.tolist(), "\tloss", loss(a2, y))
    # we have the NN graphed as
    # W1    x
    # |_____|
    #       |
    #       z1
    #       |
    # W2    a1
    # |_____|
    #       |
    #       z2
    #       |
    # y     a2
    # |_____|
    #       |
    #       C

    # we have the following partial derivatives
    # dC/dW2_jk = dz2_j/dW2_jk * da2_j/dz2_j * dC/da2_j
    # dC/da1_k = sum_j(dz2_j/da1_k * da2_j/dz2_j * dC/da2_j)
    # dC/dW1_ki = dz1_k/dW1_ki * da1_k/dz1_k * dC/da1_k

    # the terms we need to calculate are
    # dC/da2_j
    dC_da2 = - y / a2

    # da2_j/dz2_j (softmax)
    da2_dz2 = ((np.sum(z2) - 1) / np.sum(z2)) * np.exp(z2)

    # dz2_j/dW2_jk 
    dz2_dw2 = np.tile(a1, 2)

    # dz2_j/da1_k
    dz2_da1 = w2.copy()

    # da1_k/dz1_k (relu)
    da1_dz1 = z1.copy()
    da1_dz1[da1_dz1 > 0] = 1
    da1_dz1[da1_dz1 <= 0] = 0.01

    # dz1_k/dW1_ki
    dz1_dw1 = np.tile(x, 3)

    dC_da1 = dz2_da1 @ (da2_dz2 * dC_da2)

    dC_dw2 = dz2_dw2 * (da2_dz2 * dC_da2).T

    dC_dw1 = dz1_dw1 * (da1_dz1 * dC_da1).T

    w1 -= alpha * dC_dw1
    w2 -= alpha * dC_dw2
