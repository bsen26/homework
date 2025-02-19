from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
)


class CONFIG:
    batch_size = 128
    num_epochs = 20
    initial_learning_rate = 0.05
    initial_weight_decay = 0.001

    lrs_kwargs = {"cycle_length": 1955, "max_lr": 0.05, "base_lr": 0.001}

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.SGD(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
        momentum=0.9,
    )

    transforms = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
