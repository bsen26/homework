import torch.nn as nn
import torch


class Model(torch.nn.Module):
    """
    A convolutional neural network for image classification on the CIFAR10 dataset.

    Args:
        num_channels (int): Number of input channels (usually 3 for RGB images).
        num_classes (int): Number of classes in the output (usually 10 for CIFAR10).

    Methods:
        forward: Defines the forward pass of the model.

    Example:
        To create a model for CIFAR10 classification:

        model = Model(num_channels=3, num_classes=10)
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model for CIFAR10.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
