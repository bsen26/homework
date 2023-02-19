from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Creating a custom LR scheduler module
    """
    def __init__(self, optimizer, last_epoch=-1):
        """
        Create a new scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.initial_lr = 0.1  # factor by which to decrease the learning rate
        self.lr_decay_interval = 25  # number of epochs after which to decrease the learning rate
        self.steps = [80, 120, 160]
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Testing out the docstring
        """
        # compute the new learning rate
        epoch = self.last_epoch
        if epoch < self.steps[0]:
            lr = self.initial_lr
        elif epoch < self.steps[1]:
            lr = self.initial_lr * 0.1
        elif epoch < self.steps[2]:
            lr = self.initial_lr * 0.01
        else:
            lr = self.initial_lr * 0.001
        return [lr for _ in self.optimizer.param_groups]

