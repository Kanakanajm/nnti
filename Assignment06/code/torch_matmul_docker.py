#!/usr/bin/env python3
import sys
import torch
import time

print("Running python version: ", sys.version)
print("PyTorch version :", torch.__version__)
print("GPU Available: ", torch.cuda.is_available())
print("Num GPUs Available: ", torch.cuda.device_count())


matrix1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
matrix2 = torch.linalg.inv(matrix1)
product = matrix1 @ matrix2

print(product)
