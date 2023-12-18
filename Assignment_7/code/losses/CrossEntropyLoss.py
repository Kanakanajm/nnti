import numpy as np


class CrossEntropy:
    def __init__(self, class_count=None, average=True):
        self._EPS = 1e-8
        self.classes_counts = class_count
        self.average = average

    def __call__(self, Y_pred, Y_real):
        """
        expects: Y_pred - N*D matrix of predictions (N - number of datapoints)
                 Y_real - N*D matrix of one-hot vectors
        apply softmax before computing negative log likelihood loss
        return a scalar
        """
        raise NotImplementedError()

    def grad(self):
        """
        returns gradient with the size equal to the the size of the input vector (self.y_pred)
        """
        raise NotImplementedError()
