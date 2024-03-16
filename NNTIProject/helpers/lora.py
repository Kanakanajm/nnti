import torch
class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, r, a):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(r).float())
        self.A = torch.nn.Parameter(torch.randn(in_features, r) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(r, out_features))
        self.a = a

    def forward(self, x):
        x = self.a * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, r, a):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, r, a
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)