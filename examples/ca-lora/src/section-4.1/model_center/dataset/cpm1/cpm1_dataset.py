# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.utils.data as data
from ..indexed import MMapIndexedDataset
import random
import numpy as np


class CPM1_Dataset(data.Dataset):
    def __init__(self, ctx : MMapIndexedDataset, 
                       tgt : MMapIndexedDataset,
                       max_length = 1024):
        self.ctx = ctx
        self.tgt = tgt
        self.max_length = max_length

    def __len__(self):
        return len(self.ctx)
    
    def __get_item_data(self, ctx, tgt):
        if ctx.shape[0] > self.max_length or tgt.shape[0] > self.max_length:
            return None, None, None
        assert len(ctx) == len(tgt)
        len_ctx = min(ctx.shape[0], self.max_length)

        ctx = ctx.astype('int64')
        tgt = tgt.astype('int64')

        th_ctx = torch.zeros(self.max_length, dtype=torch.long)
        th_ctx[:len_ctx] = torch.from_numpy(ctx)[:len_ctx].long()
        th_tgt = torch.full((self.max_length,), -100, dtype=torch.long)
        th_tgt[:len_ctx] = torch.from_numpy(tgt)[:len_ctx].long()
        return th_ctx, len_ctx, th_tgt

    def __getitem__(self, index):
        ctx = self.ctx[index]
        tgt = self.tgt[index]

        if isinstance(index, int):
            th_ctx, len_ctx, th_tgt = self.__get_item_data(ctx, tgt)
            return {
                "ctx": th_ctx,
                "tgt": th_tgt,
                "len_ctx": len_ctx,
            }
        else:
            res = {"ctx": [], "tgt": [], "len_ctx": [],}
            for _ctx, _tgt in zip(ctx, tgt):
                _th_ctx, _len_ctx, _th_tgt = self.__get_item_data(_ctx, _tgt)
                if _th_ctx is None:
                    continue
                res["ctx"].append(_th_ctx)
                res["tgt"].append(_th_tgt)
                res["len_ctx"].append(_len_ctx)
            return {
                "ctx": torch.stack(res["ctx"]), 
                "tgt": torch.stack(res["tgt"]),
                "len_ctx": torch.LongTensor(res["len_ctx"]),
            }


class CPM1_Dataset_Merge(data.Dataset):
    def __init__(self, ctx : MMapIndexedDataset, max_length = 1024):
        self.ctx = ctx
        self.max_length = max_length

    def __len__(self):
        return len(self.ctx)
    
    def __get_item_data(self, ctx):
        if ctx.shape[0] > self.max_length:
            return None, None, None, None
        len_ctx = min(ctx.shape[0], self.max_length)
        lef = random.randint(len_ctx // 8, len_ctx // 4)
        rig = random.randint(len_ctx // 4 * 3, len_ctx)
        if ctx[len_ctx-1] == 4:
            rig = len_ctx
        tgt = np.full((len_ctx), -100)
        tgt[lef-1:rig-1] = ctx[lef:rig]
        context_ctx = np.arange((len_ctx))
        context_ctx = (context_ctx < lef) | (context_ctx >= rig)
        return ctx, tgt, len_ctx, context_ctx

    def __getitem__(self, index):
        ctx = self.ctx[index]
        th_ctx, th_tgt, len_ctx, context_ctx = self.__get_item_data(ctx)
        return th_ctx, th_tgt, len_ctx, context_ctx
