import numpy as np


class Sigmoid:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        raise NotImplementedError()

    def get_type(self):
        return "activation"

    def grad(self, in_gradient):
        raise NotImplementedError()
