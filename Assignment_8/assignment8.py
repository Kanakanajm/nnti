from typing import Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
        raise NotImplementedError

    def forward(self, images):
        # Your code here
        raise NotImplementedError


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
    
    # Your code here
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

    return model