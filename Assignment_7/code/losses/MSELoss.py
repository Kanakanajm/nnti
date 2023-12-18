import numpy as np


class MSELoss:
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, y_pred):
        # save the inputs
        raise NotImplementedError()

    def grad(self):
        """
        returns gradient equal to the the size of input vector (self.y_pred)
        """
        raise NotImplementedError()
