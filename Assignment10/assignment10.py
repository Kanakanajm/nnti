from typing import Optional, Callable

import torch
from torchvision.datasets import GTSRB
from torchvision.transforms import v2

# my imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    train_set = GTSRB(root="./data", split="train", download=False, transform=transform)
    test_set = GTSRB(root="./data", split="test", download=False, transform=transform)
    return train_set, test_set


# Implement your CNN and finetune ResNet18
# Don't forget to submit your loss and accuracy results in terms of the log file.
batch_size = 4

train_set, test_set = get_data_split()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel size = 5
        # conv1's output will have shape 6*28*28
        self.conv1 = nn.Conv2d(3, 6, 5)
        # conv2's output will have shape 12*24*24
        self.conv2 = nn.Conv2d(6, 12, 5)

        # GTSRB dataset has 43 classes
        # the labels of the data provided are one hot encoded (i.e. single integer representing class number)
        # but the output layer of the network should have 43 nodes representing classes.
        self.fc1 = nn.Linear(12 * 24 * 24, 3 * 24 * 24)
        self.fc2 = nn.Linear(3 * 24 * 24, 43)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

         # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

# from here on, most of the lines are taken from the CIFAR10 pytorch tutorial
# see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net.train()
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # print loss at every 2000 iterations
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
