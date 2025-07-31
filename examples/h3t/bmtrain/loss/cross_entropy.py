from typing import Optional
import torch
from . import _cuda as C

class OpFusedCrossEntropy(torch.autograd.Function):
    """
    CrossEntropy dim = 1
    """
    @staticmethod
    def forward(ctx, x : torch.Tensor, target : torch.Tensor, ignore_index: int):
        assert x.ndim == 2
        softmax = torch.empty(x.size(), device=x.device, dtype=x.dtype)
        out = torch.empty(x.size(0), device=x.device, dtype=torch.float)
        C.f_cross_entropy_forward(
            x.size(0), x.size(1),
            x, target,
            softmax, out,
            ignore_index,
        )
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(softmax, target)
        return out # float tensor
        
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        grad_output = grad_output.contiguous()
        softmax, target = ctx.saved_tensors
        C.f_cross_entropy_backward_inplace(
            softmax.size(0), softmax.size(1),
            grad_output, target,
            softmax,
            ctx.ignore_index,
        )
        return (softmax, None, None)

class OpFusedCrossEntropyInplace(torch.autograd.Function):
    """
    CrossEntropy dim = 1
    """
    @staticmethod
    def forward(ctx, x : torch.Tensor, target : torch.Tensor, ignore_index: int):
        assert x.ndim == 2
        out = torch.empty(x.size(0), device=x.device, dtype=torch.float)
        C.f_cross_entropy_forward_inplace(
            x.size(0), x.size(1),
            x, target,
            out,
            ignore_index,
        ) # x is inplace modify to softmax result
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(x, target)
        return out # float tensor
        
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        grad_output = grad_output.contiguous()
        softmax, target = ctx.saved_tensors
        C.f_cross_entropy_backward_inplace(
            softmax.size(0), softmax.size(1),
            grad_output, target,
            softmax,
            ctx.ignore_index,
        ) # softmax is inplace modify to grad_input
        return (softmax, None, None)

class FusedCrossEntropy(torch.nn.Module):
    r"""This criterion computes the cross entropy loss between input and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.
    `input` has to be a Tensor of size :math:`(minibatch, C)`.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C-1]` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
      :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\exp(\sum_{i=1}^C x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`.
        - Target: If containing class indices, shape :math:`(N)` where each value is
          :math:`0 \leq \text{targets}[i] \leq C-1`. If containing class probabilities,
          same shape as the input.
        - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N)`.
          Otherwise, scalar.

    Examples::

        >>> # Example of target with class indices
        >>> loss_func = bmt.loss.FusedCrossEntropy()
        >>> input = torch.randn(32, 100).half()
        >>> target = torch.randint(0, 100, (32,)).long()
        >>> loss = loss_func(input, target)
        >>> loss.backward()
    """
    def __init__(self,
                 weight: Optional[torch.Tensor] = None,
                 ignore_index: int = -100,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0, # TODO not supported yet
                 inplace: bool = False,
                ) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace = inplace

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            ret = OpFusedCrossEntropyInplace.apply(input, target.int(), self.ignore_index) # return float tensor
        else:
            ret = OpFusedCrossEntropy.apply(input, target.int(), self.ignore_index) # return float tensor

        if self.weight is not None:
            if self.weight.dim() != 1 or self.weight.size(0) != input.size(1):
                raise ValueError("weight should be a 1D tensor of size C");
            w = self.weight[torch.where(target==self.ignore_index, 0, target)].float()
            w[target==self.ignore_index] = 0
        else:
            w = (target != self.ignore_index).int()

        ret = w * ret
        
        if self.reduction == "none":
            return ret
        elif self.reduction == "sum":
            return ret.sum()
        elif self.reduction == "mean":
            return ret.sum() / w.sum().float()
