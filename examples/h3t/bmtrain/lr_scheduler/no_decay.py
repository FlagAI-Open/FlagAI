

from .warmup import WarmupLRScheduler

class NoDecay(WarmupLRScheduler):
    r"""
        After a warmup period during which learning rate increases linearly between 0 and the start_lr,
        The decay period performs :math:`\text{lr}=\text{start_lr}`
    """
    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter
    
    def get_lr_decay(self, num_iter) -> float:
        return self.start_lr


        
   