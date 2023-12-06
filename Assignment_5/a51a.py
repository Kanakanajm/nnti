import numpy as np

A = 1
B = 100
# E = 0.002
E = 0.0001
X0 = np.array([0.9, 1.12])

def f(x):
    return (A - x[0])**2 + B*(x[1] - x[0]**2)**2

def gradient(x):
    return np.array([
        -2*(A - x[0]) - 4*B*x[0]*(x[1] - x[0]**2),
        2*B*(x[1] - x[0]**2)
    ])

def about(x):
    print("X:", x, "\tgrad:", gradient(x), "\tf:", f(x))

X = X0
NO_ITERATION = 3
GRAD_THRESHOLD = 0.01

i = 0
while(i < NO_ITERATION and np.linalg.norm(gradient(X)) > GRAD_THRESHOLD):
    print("iteration", i+1)
    about(X)

    X = X - E * gradient(X)
    i += 1

print("result")
about(X)
