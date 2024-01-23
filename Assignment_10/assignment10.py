from typing import Optional, Callable

import torch
from torchvision.datasets import GTSRB
from torchvision.transforms import v2

# You may add aditional augmentations, but don't change the output size
_resize_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((32, 32))
])


def get_data_split(transform: Optional[Callable] = _resize_transform):
    """
    Downloads and returns the test and train set of the German Traffic Sign Recognition Benchmark (GTSRB)
    dataset.

    :param transform: An optional transform applied to the images
    :returns: Train and test Dataset instance
    """
    train_set = GTSRB(root="./data", split="train", download=True, transform=transform)
    test_set = GTSRB(root="./data", split="test", download=True, transform=transform)
    return train_set, test_set


# Implement your CNN and finetune ResNet18
# Don't forget to submit your loss and accuracy results in terms of the log file.