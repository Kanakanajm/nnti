import numpy as np
import copy


class Dropout:
    def __init__(self, layer, p: float = 0.5):
        pass

    def __call__(self, x):
        """
        apply inverted dropout
        """
        raise NotImplementedError()

    def get_type(self):
        return "layer"

    def grad(self, in_gradient):
        raise NotImplementedError()
