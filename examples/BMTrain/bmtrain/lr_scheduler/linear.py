from .warmup import WarmupLRScheduler


class Linear(WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{\text{end_iter}-\text{num_iter}}{\text{end_iter}-\text{warmup_iter}}`
    """

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        return max(
            0.0,
            self.start_lr
            * (self.end_iter - num_iter)
            / (self.end_iter - self.warmup_iter),
        )
