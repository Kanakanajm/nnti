from typing import Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy
 


# 8.4.1
class FFNN(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        """
        A simple feed forward neural network with one hidden layer
        
        :params input_shape: Number of features
        :params hidden_units: Number of neurons in the hidden layer
        :params output_shape: Number of output units / classes
        """
        # Your code here
        self.linear1 = nn.Linear(input_shape, hidden_units)
        self.bnorm = nn.BatchNorm1d(hidden_units)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_units, output_shape)

    def forward(self, images):
        # Your code here
        z = self.linear1(images)
        z_normed = self.bnorm(z)
        a = self.relu(z_normed)
        out = self.linear2(a)
        return out


# 8.4.2
def train(model: nn.Module, optimizer: torch.optim.Optimizer, criterion, dataloader: DataLoader) -> Tuple[float, float]:
    """
    Trains a model for one epoch on the full dataloader.

    :param model: The neural network model to be trained.
    :param optimizer: The optimization algorithm used for training.
    :param criterion: The loss function used to evaluate the model's performance.
    :param dataloader: DataLoader containing the training dataset.

    :return: A tuple containing the average loss and average accuracy over the dataset.
    """
    # switch the mode of the model
    model.train()
    # Your code here
    loss_sum = 0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(dataloader):
        # device has to be the same as model
        images.to("cuda" if torch.cuda.is_available() else "cpu")
        labels.to("cuda" if torch.cuda.is_available() else "cpu")


        images = images.view(-1, 28 * 28)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, torch.eye(10)[labels])
        loss_sum += loss.item()

        loss.backward()
        optimizer.step()
        total += labels.shape[0]
        for bi in range(outputs.shape[0]):
            if (torch.argmax(outputs[bi, :]) == labels[bi]):
                correct += 1

    return loss_sum / len(iter(dataloader)), correct / total

    


@torch.no_grad()
def validate(model: nn.Module, criterion, dataloader: DataLoader) -> Tuple[float, float]:
    """
    Evaluates a model on the full dataloader.

    :param model: The neural network model to be trained.
    :param criterion: The loss function used to evaluate the model's performance.
    :param dataloader: DataLoader containing the training dataset.

    :return: A tuple containing the average loss and average accuracy over the dataset.
    """
    # Your code here
    # somehow has to switch mode for validation
    model.eval()

    loss_sum = 0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images.to("cuda" if torch.cuda.is_available() else "cpu")
        labels.to("cuda" if torch.cuda.is_available() else "cpu")


        images = images.view(-1, 28 * 28)
        outputs = model(images)

        loss = criterion(outputs, torch.eye(10)[labels])
        loss_sum += loss.item()

        total += labels.shape[0]
        for bi in range(outputs.shape[0]):
            if (torch.argmax(outputs[bi, :]) == labels[bi]):
                correct += 1

    return loss_sum / len(iter(dataloader)), correct / total

# 8.4.3
def training_loop(
    train_set, 
    test_set, 
    epochs: int, 
    batch_size: int,
    learning_rate: float, 
    weight_decay: float = 0.0, 
    early_stopping_patience: Optional[int] = None,
):
    """
    Fully trains the netwok for at most `epoch` epochs. 
    If early_stopping_patience > 0, training should be stopped if the validation loss has not
    decreased in the last `early_stopping_patience` epochs.

    Should return the best model and validation loss of that model.

    :params train_set: Training dataset
    :params test_set: Test dataset
    :params epochs: Number of training epochs
    :params batch_size: Number of samples per batch
    :params learning_rate: Step size of the optimizer
    :params weight_decay: Regularization strength
    :params early_stopping_patience: After how many non-improving epochs to stop
    :returns: Best model and validation loss of that model
    """
    torch.manual_seed(13415)    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    model = FFNN(input_shape=28*28, hidden_units=196, output_shape=10)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Your code here
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

    train_history = []
    val_history = []
    best_accuracy = 0
    best_model = None
    bad_epoch = 0
    for epoch in range(int(epochs)):
        train_loss, train_accu = train(model, optimizer, criterion, train_loader)
        val_loss, val_accu = validate(model, criterion, test_loader)
        if (val_accu > best_accuracy): 
            best_accuracy = val_accu
            best_model = copy.deepcopy(model)

        # check progress if early_stopping_patience is positive
        if (early_stopping_patience > 0):
            # comparing with the last validation accuracy
            if (epoch > 0 and val_history[-1][1] >= val_accu):
                bad_epoch += 1
                if (bad_epoch > early_stopping_patience):
                    print("Early stopped at", epoch)
                    break
            else:
                bad_epoch = 0
            
        train_history.append((train_loss, train_accu))
        val_history.append((val_loss, val_accu))
        

    


    # plot stuff
    plt.plot(range(len(train_history)), [train_history[i][0] for i in range(len(train_history))], label="train loss")
    plt.plot(range(len(train_history)), [train_history[i][1] for i in range(len(train_history))], label="train accuracy")
    plt.plot(range(len(val_history)), [val_history[i][0] for i in range(len(val_history))], label="validation loss")
    plt.plot(range(len(val_history)), [val_history[i][1] for i in range(len(val_history))], label="validation accuracy")
    plt.legend()
    plt.show()

    print("Best Accuracy:", best_accuracy)
    return best_model