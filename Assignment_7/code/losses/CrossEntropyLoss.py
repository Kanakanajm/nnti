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
        self.Y_pred = Y_pred
        self.Y_real = Y_real
        # To avoid log(0), we add a small epsilon value to predictions
        predictions = np.clip(CrossEntropy.softmax(Y_pred), self._EPS, 1.0)
        log_predictions = np.log(predictions)
        loss = -np.sum(Y_real * log_predictions, axis=1)

        if self.average:
            loss = np.mean(loss)

        return loss

    def grad(self):
        """
        returns gradient with the size equal to the the size of the input vector (self.y_pred)
        """
        # To avoid log(0), we add a small epsilon value to predictions
        predictions = np.clip(CrossEntropy.softmax(self.Y_pred), self._EPS, 1.0)
        return (predictions - self.Y_real) / predictions.shape[0]

    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    @classmethod
    def softmax(self, x):
        """
        return softmax values for each sets of scores in x (row-wise)
        every sample is a row in x
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1).reshape(-1, 1)
