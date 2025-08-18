from .warmup import WarmupLRScheduler


class Exponential(WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \gamma ^ {\left(\text{num_iter}-\text{warmup_iter}\right)}`
    """

    def __init__(
        self, optimizer, start_lr, warmup_iter, end_iter, num_iter, gamma=0.95
    ) -> None:
        super().__init__(optimizer, start_lr, warmup_iter, end_iter, num_iter)
        self.gamma = gamma

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        return max(0.0, self.start_lr * self.gamma ** (num_iter - self.warmup_iter))
