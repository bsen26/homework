from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomCrop,
    ToTensor,
    Normalize,
)


class CONFIG:
    batch_size = 8
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)

    transforms = Compose(
        [
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
