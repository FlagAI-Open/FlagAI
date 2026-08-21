import math
from .warmup import WarmupLRScheduler


class Cosine(WarmupLRScheduler):
    r"""
        After a warmup period during which learning rate increases linearly between 0 and the start_lr,
        The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """
    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        return max(0.0, self.start_lr * 0.5 * (1.0 + math.cos(progress * math.pi)))
