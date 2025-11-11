# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRFLayer(nn.Module):
    """
    """

    def __init__(self, output_dim):
        super(CRFLayer, self).__init__()

        self.output_dim = output_dim
        self.trans = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.trans.data.uniform_(-0.1, 0.1)

    def compute_loss(self, y_pred, y_true, mask):
        """
        计算CRF损失
        """
        y_pred = y_pred * mask
        y_true = y_true * mask
        target_score = self.target_score(y_pred, y_true)
        log_norm = self.log_norm_step(y_pred, mask)
        log_norm = self.logsumexp(log_norm, dim=1)  # 计算标量
        return log_norm - target_score

    def forward(self, y_pred, y_true, mask):
        """
        y_true: [[1, 2, 3], [2, 3, 0] ]
        mask: [[1, 1, 1], [1, 1, 0]]
        """
        if y_pred.shape[0] != mask.shape[0] or y_pred.shape[1] != mask.shape[1]:
            raise Exception("mask shape is not match to y_pred shape")
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        mask = mask.float()
        y_true = y_true.reshape(y_pred.shape[:-1])
        y_true = y_true.long()
        y_true_onehot = F.one_hot(y_true, self.output_dim)
        y_true_onehot = y_true_onehot.float()

        return self.compute_loss(y_pred, y_true_onehot, mask)

    def target_score(self, y_pred, y_true):
        """
        计算状态标签得分 + 转移标签得分
        y_true: (batch, seq_len, out_dim)
        y_pred: (batch, seq_len, out_dim)
        """
        # print(y_pred.shape)
        # print(y_true.shape)
        point_score = torch.einsum("bni,bni->b", y_pred, y_true)
        trans_score = torch.einsum("bni,ij,bnj->b", y_true[:, :-1], self.trans,
                                   y_true[:, 1:])

        return point_score + trans_score

    def log_norm_step(self, y_pred, mask):
        """
        计算归一化因子Z(X)
        """
        state = y_pred[:, 0]  # 初始Z(X)
        y_pred = y_pred[:, 1:].contiguous()
        mask = mask[:, 1:].contiguous()
        batch, seq_len, out_dim = y_pred.shape
        for t in range(seq_len):
            cur_mask = mask[:, t]
            state = torch.unsqueeze(state, 2)  # (batch, out_dim, 1)
            g = torch.unsqueeze(self.trans, 0)  # (1, out_dim, out_dim)
            outputs = self.logsumexp(state + g, dim=1)  # batch, out_dim
            outputs = outputs + y_pred[:, t]
            outputs = cur_mask * outputs + (1 - cur_mask) * state.squeeze(-1)
            state = outputs

        return outputs

    def logsumexp(self, x, dim=None, keepdim=False):
        """
        避免溢出
        """
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        out = xm + torch.log(
            torch.sum(torch.exp(x - xm), dim=dim, keepdim=True))
        return out if keepdim else out.squeeze(dim)
