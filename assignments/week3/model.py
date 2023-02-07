import torch
import torch.nn as nn
from typing import Callable


class MLP(torch.nn.Module):
    """
    This is the Multi-Layer Perceptron class
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.input = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(hidden_count)]
        )
        self.output = nn.Linear(hidden_size, num_classes)
        self.actv = activation()

        initializer(self.input.weight)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.float32) -> torch.float32:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = torch.flatten(x, start_dim=1)
        x = self.actv(self.input(x))
        x = self.dropout(x)
        for layer in self.linears:
            x = layer(x)
            x = self.actv(x)
            x = self.dropout(x)
        x = self.output(x)
        return x
