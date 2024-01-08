# This file builds upon your implementations in the assignment 8 file.
# You have seen the validation loss and accuracy for two model configurations for now.
# In this file we will do a grid search over hyperparameters
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset

from assignment8 import training_loop


# 8.5
def hyperparameter_search(train_set, test_set):
    # Your code here
    raise NotImplementedError


    


# Keep as is
def reduce_dataset(dataset, rng, num_samples=100) -> torch.Tensor:
    """
    Reduces the MNIST dataset to have only num_samples for each class.
    Returns the indices of all datapoints that should be kept
    """
    reduced_indices = []
    for target in range(10):
        mask = dataset.targets == target
        indices = np.nonzero(mask).flatten()
        random_subset = rng.choice(indices, size=num_samples, replace=False)
        reduced_indices.extend(random_subset)

    return reduced_indices


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    normalized_mnist = MNIST(root="./data", download=True, train=True, transform=transform)
    tiny_mnist_indices = reduce_dataset(normalized_mnist, rng=rng, num_samples=100)
    standard_train_set = Subset(normalized_mnist, tiny_mnist_indices)
    test_set = MNIST(root="./data", download=True, train=False, transform=transform)


    hyperparameter_search(standard_train_set, test_set)