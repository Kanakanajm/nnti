import numpy as np
from typing import List

def init_weights(row_size: int, col_size: int) -> np.ndarray:
    return np.random.randn(row_size, col_size)

def init_bias(size: int) -> np.ndarray:
    return np.random.randn(size,)
def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class RNN():
    def __init__(self, input_size: int, hidden_size: int):
        self.y: List[np.ndarray] = None
        self.h: List[np.ndarray] = None

        self.W: np.ndarray = init_weights(hidden_size, hidden_size)
        self.U: np.ndarray = init_weights(hidden_size, input_size)
        self.V: np.ndarray = init_weights(input_size, hidden_size)

        self.b: np.ndarray = init_bias(hidden_size)
        self.c: np.ndarray = init_bias(input_size)

    def init_h(self, h: np.ndarray):
        self.h = [h]

    def forward(self, x, t = 1):
        self.h.append(np.tanh(np.dot(self.W, self.h[t-1]) + np.dot(self.U, x) + self.b))
        self.y.append(softmax(np.dot(self.V, self.h[t]) + self.c))
        
    def __call__(self, X: np.ndarray, h0: np.ndarray):
        self.y = []
        self.init_h(h0)
        for i in range(X.shape[0]):
            self.forward(X[i, :], i+1)
        return self.y
        
    
# rnn = RNN(10, 20)

# rnn.init_h(np.random.randn(20,))
# x = np.ones((10,))

# for _ in range(10):
#     y = rnn.forward(x)
#     print(y)