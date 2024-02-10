import numpy as np


class MSELoss:
    def __init__(self) -> None:
        self.y_true = None
        self.y_pred = None
        self.n = None

    def __call__(self, y_true, y_pred):
        # save the inputs
        self.y_true = y_true
        self.y_pred = y_pred
        self.n = y_true.shape[0]
        return 1 / self.n * np.sum((y_true - y_pred) ** 2)

    def grad(self):
        """
        returns gradient equal to the the size of input vector (self.y_pred)
        """
        return 2 / self.n * (self.y_pred - self.y_true)
