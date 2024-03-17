import torch

class IA3Layer(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.v = torch.nn.Parameter(torch.randn((1, size)))
    def forward(self, x):
        x = self.v * x
        return x

class LinearWithIA3(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.ia3 = IA3Layer(
            linear.out_features
        )

    def forward(self, x):
        return self.ia3(self.linear(x))

class ActivationWithIA3(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.ia3 = IA3Layer(
            activation.out_features
        )

    def forward(self, x):
        return self.ia3(self.activation(x))