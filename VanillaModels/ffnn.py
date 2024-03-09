import numpy as np

def relu(x: np.ndarray):
    return np.maximum(x, 0, x)
def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class SingleLayerFeedForwardNeuralNetwork():

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.W = np.random.randn(hidden_dim, input_dim)
        self.b1 = np.random.randn(hidden_dim, 1)
        self.U = np.random.randn(output_dim, hidden_dim)
        self.b2 = np.random.randn(output_dim, 1)
    
    def forward(self, input: np.ndarray):
        self.x = input
        self.z = self.W @ self.x + self.b1
        self.h = relu(self.z)
        self.θ = self.U @ self.h + self.b2
        self.ŷ = softmax(self.θ)

        return self.ŷ
    
    def backward(self, y, lr):
        J = -np.sum(y * np.log(self.ŷ))

        δ1: np.ndarray = (y - self.ŷ).T
        δ2 = δ1 @ self.U @ np.diag(np.sign(self.h).flatten())

        dJ_dU = δ1.T @ self.h.T
        dJ_db2 = δ1.T
        dJ_dW = δ2.T @ self.x.T
        dJ_db1 = δ2.T
        
        self.W -= lr * dJ_dW
        self.b1 -= lr * dJ_db1
        self.U -= lr * dJ_dU
        self.b2 -= lr * dJ_db2

        return J
    

# test it out
    
# define a function to proximate
def f(x):
    if (x[0, 0] * x[1, 0] * x[2, 0]) > 0:
        return np.array([[1], [0]])
    
    return np.array([[0], [1]])

# batch_size = 64
# create a model
model = SingleLayerFeedForwardNeuralNetwork(3, 100, 2)
for i in range(10000000):
    x = np.random.randn(3, 1)
    y = f(x)
    model.forward(x)
    J = model.backward(y, 0.0001)
    print(J)

# test the model
x = np.array([[1], [-9], [66]])
print(f(x))
print(model.forward(x))
