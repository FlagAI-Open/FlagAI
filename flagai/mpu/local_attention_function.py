# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from localAttention import (similar_forward, similar_backward,
                            weighting_forward, weighting_backward_ori,
                            weighting_backward_weight)

__all__ = ['f_similar', 'f_weighting']


class similarFunction(Function):

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW, casual_mask=False):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        ctx.casual_mask = casual_mask
        output = similar_forward(x_ori, x_loc, kH, kW, casual_mask)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        casual_mask = ctx.casual_mask
        grad_outputs = grad_outputs.contiguous()
        grad_ori = similar_backward(x_ori, x_loc, grad_outputs, kH, kW, True,
                                    casual_mask)
        grad_loc = similar_backward(x_ori, x_loc, grad_outputs, kH, kW, False,
                                    casual_mask)

        return grad_ori, grad_loc, None, None, None


class weightingFunction(Function):

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW, casual_mask=False):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        ctx.casual_mask = casual_mask
        output = weighting_forward(x_ori, x_weight, kH, kW, casual_mask)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        casual_mask = ctx.casual_mask
        grad_outputs = grad_outputs.contiguous()
        grad_ori = weighting_backward_ori(x_ori, x_weight, grad_outputs, kH,
                                          kW, casual_mask)
        grad_weight = weighting_backward_weight(x_ori, x_weight, grad_outputs,
                                                kH, kW, casual_mask)

        return grad_ori, grad_weight, None, None, None


f_similar = similarFunction.apply
f_weighting = weightingFunction.apply
