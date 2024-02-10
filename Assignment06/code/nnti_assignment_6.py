import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

"""
Team #25:
Camilo Martínez 7057573, cama00005@stud.uni-saarland.de
Honglu Ma 7055053, homa00001@stud.uni-saarland.de
"""

# Template taken from: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html?


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_loss(outputs, labels, loss_fn=torch.nn.CrossEntropyLoss()):
    return loss_fn(outputs, labels)


def train_one_epoch(
    epoch_index: int,
    model: nn.Module,
    optimizer: torch.optim,
    batch_interval_print: int = 1000,
):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = compute_loss(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if batch_interval_print != -1 and i % batch_interval_print == (
            batch_interval_print - 1
        ):
            last_loss = running_loss / 1000  # loss per batch
            tb_x = epoch_index * len(training_loader) + i + 1
            print(f"\t\tbatch {i + 1} loss: {last_loss} Loss/train: {tb_x}")
            running_loss = 0.0

    return last_loss


def perform_training(
    model: nn.Module,
    optimizer: torch.optim,
    epochs: int,
    batch_interval_print: int = 1000,
    save_folder_path: str = None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_number = 0

    best_vloss = 1_000_000.0

    for epoch in range(epochs):
        print("\n\tEpoch {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, model, optimizer, batch_interval_print)

        running_vloss = 0.0

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = compute_loss(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        # Log the running loss averaged per batch
        # for both training and validation
        print(
            f"\t\tRunning Losses = Training: {avg_loss}, Validation: {avg_vloss}, Epoch: {epoch_number + 1}"
        )

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        if save_folder_path:
            model_path = "model_{}_epoch_{}".format(timestamp, epoch_number + 1)
            torch.save(model.state_dict(), os.path.join(save_folder_path, model_path))

        epoch_number += 1


def train_val_test_split(
    dataset: torchvision.datasets,
    train_validation_split: int = 0.9,
    verbose: bool = True,
) -> tuple:
    """Returns the dataloaders for training, validation and test set from a torch dataset

    Parameters
    ----------
    train_validation_split : int, optional
        Defines how much of the original training set is used to train, the rest is used
        for the validation set, by default 0.9
    verbose : bool, optional
        If true, prints useful information, by default True

    Returns
    -------
    tuple
        training, validation and test dataloaders
    """
    # Transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the full training dataset
    full_training_data = dataset(
        "./data", train=True, transform=transform, download=True
    )

    # Splitting the full training dataset into training and validation sets
    train_size = int(
        train_validation_split * len(full_training_data)
    )  # train_validation_split % of the dataset
    validation_size = len(full_training_data) - train_size  # Remaining % of the dataset
    training_set, validation_set = random_split(
        full_training_data, [train_size, validation_size]
    )

    # Load the test dataset
    test_set = torchvision.datasets.FashionMNIST(
        "./data", train=False, transform=transform, download=True
    )

    # Data loaders for our datasets
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=4, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=4, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    # Report split sizes
    if verbose:
        print("Training set has {} instances".format(len(training_set)))
        print("Validation set has {} instances".format(len(validation_set)))
        print("Test set has {} instances".format(len(test_set)))

    return training_loader, validation_loader, test_loader


def calculate_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    """Function to calculate the accuracy of a given PyTorch model using a specified dataloader.

    Parameters
    ----------
    model : nn.Module
         Trained PyTorch model
    loader : torch.utils.data.DataLoader
        DataLoader for which the accuracy is to be calculated

    Returns
    -------
    float
        Accuracy of the model on the provided dataloader
    """
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


training_loader, validation_loader, test_loader = train_val_test_split(
    dataset=torchvision.datasets.FashionMNIST, train_validation_split=0.9
)

model = FashionMNISTModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("\nTraining FashionMNIST Model")

models_folder = os.path.join(os.getcwd(), "models")
perform_training(model, optimizer, epochs=10, save_folder_path=models_folder)

# Report the accuracies and find the best model according to validation set
print("\nReporting:")
best_validation_accuracy = 0
best_model = None
best_saved_model_file = None
best_training_accuracy = 0
best_testing_accuracy = 0
for saved_model_file in os.listdir(models_folder):
    # Load the model
    saved_model = FashionMNISTModel()
    saved_model.load_state_dict(
        torch.load(os.path.join(models_folder, saved_model_file))
    )

    # Calculate accuracies
    training_accuracy = calculate_accuracy(saved_model, training_loader)
    validation_accuracy = calculate_accuracy(saved_model, validation_loader)
    test_accuracy = calculate_accuracy(saved_model, test_loader)

    if best_validation_accuracy < validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_training_accuracy = training_accuracy
        best_testing_accuracy = test_accuracy
        best_model = saved_model
        best_saved_model_file = saved_model_file

    # Print accuracies of "in-between" models
    print(f"\nModel {saved_model_file}:")
    print(f"\tTraining Accuracy: {training_accuracy:.2f}%")
    print(f"\tValidation Accuracy: {validation_accuracy:.2f}%")
    print(f"\tTest Accuracy: {test_accuracy:.2f}%")

# Print accuracy of best trained model
print(f"\nBest found model: {best_saved_model_file}")
print(f"\tTraining Accuracy: {best_training_accuracy:.2f}%")
print(f"\tValidation Accuracy: {best_validation_accuracy:.2f}%")
print(f"\tTest Accuracy: {best_testing_accuracy:.2f}%")
