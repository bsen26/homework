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
        max_lr: float = 0.1,
        cycle_length: int = 1000,
        base_lr: float = 0.001,
        last_epoch=-1,
        **kwargs
    ):
        """
        Create a new scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use.
            max_lr (float): The maximum learning rate to use.
            cycle_length (int): The length of the cycle, in iterations.
            base_lr (float): The minimum learning rate to use.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute the new learning rate using the triangular policy.
        """
        cycle = math.floor(1 + self.last_epoch / (2 * self.cycle_length))
        x = abs(self.last_epoch / self.cycle_length - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))

        return [lr for _ in self.optimizer.param_groups]
