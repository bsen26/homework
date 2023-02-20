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
        max_epochs=25,
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
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        The get lr method is used to compute the new learning rate, in this case we are using cosine annealing
        """
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
