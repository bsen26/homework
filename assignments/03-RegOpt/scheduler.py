from typing import List
import math
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Creating a custom LR scheduler module
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs=5,
        max_epochs=100,
        warmup_factor=0.1,
        last_epoch=-1,
        **kwargs
    ):
        """
        Create a new scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        # self.initial_lr = 0.1  # factor by which to decrease the learning rate
        # self.lr_decay_interval = 25  # number of epochs after which to decrease the learning rate
        # self.steps = [80, 120, 160]
        # super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Testing out the docstring
        """
        # compute the new learning rate
        # epoch = self.last_epoch
        # if epoch < self.steps[0]:
        #     lr = self.initial_lr
        # elif epoch < self.steps[1]:
        #     lr = self.initial_lr * 0.1
        # elif epoch < self.steps[2]:
        #     lr = self.initial_lr * 0.01
        # else:
        #     lr = self.initial_lr * 0.001
        # return [lr for _ in self.optimizer.param_groups]

        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * self.warmup_factor * (self.last_epoch + 1)
                for base_lr in self.base_lrs
            ]
        else:
            factor = 0.5 * (
                1
                + math.cos(
                    (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                    * math.pi
                )
            )
            return [base_lr * factor for base_lr in self.base_lrs]
