import torch

class WarmupLRScheduler:
    r"""Base class for learning rate schedulers with warmup.

    Args:
        optimizer (torch.optim.Optimizer): optimizer used for training
        start_lr (float): starting learning rate
        warmup_iter (int): number of iterations to linearly increase learning rate
        end_iter (int): number of iterations to stop training
        num_iter (int): current iteration number
    """
    
    def __init__(self, optimizer : torch.optim.Optimizer, start_lr, warmup_iter, end_iter, num_iter=0) -> None:
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter
        self.optimizer = optimizer
        self.num_iter = num_iter
        self._current_lr = None
        
        self.step(self.num_iter)
    
    def get_lr_warmup(self, num_iter) -> float: ...

    def get_lr_decay(self, num_iter) -> float: ...

    def get_lr(self):
        assert self.num_iter >= 0
        
        if self.num_iter < self.warmup_iter:
            return self.get_lr_warmup(self.num_iter)
        else:
            return self.get_lr_decay(self.num_iter)

    @property
    def current_lr(self):
        return self._current_lr

    def step(self, num_iter = None) -> None:
        if num_iter is None:
            num_iter = self.num_iter + 1
        self.num_iter = num_iter

        lr = self.get_lr()
        self._current_lr = lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        
    def state_dict(self):
        return {
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "end_iter": self.end_iter,
            "num_iter": self.num_iter
        }

    def load_state_dict(self, state_dict):
        self.start_lr = state_dict["start_lr"]
        self.warmup_iter = state_dict["warmup_iter"]
        self.end_iter = state_dict["end_iter"]
        self.num_iter = state_dict["num_iter"]

        self.step(self.num_iter)
        
