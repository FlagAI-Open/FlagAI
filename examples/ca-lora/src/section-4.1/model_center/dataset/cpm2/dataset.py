import torch
import torch.utils.data as data
from ..indexed import MMapIndexedDataset
import random
import numpy as np

class CPM2_Dataset(data.Dataset):
    def __init__(self, ctx : MMapIndexedDataset, 
                       tgt : MMapIndexedDataset,
                       max_source_length = 512,
                       max_target_length = 256):
        self.ctx = ctx
        self.tgt = tgt
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

    def __len__(self):
        return len(self.ctx)
    
    def __get_item_data(self, ctx, tgt):
        # TODO 26240
        ctx = ctx - (ctx >= 26240) * 190
        tgt = tgt - (tgt >= 26240) * 190

        if ctx.shape[0] > self.max_source_length or tgt.shape[0] > self.max_target_length+1: # TODO
            return None, None, None, None
        len_ctx = min(ctx.shape[0], self.max_source_length)
        len_tgt = min(tgt.shape[0], self.max_target_length)

        # TODO
        # ctx.astype('int64')
        # tgt.astype('int64')

        th_ctx = torch.zeros(self.max_source_length, dtype=torch.long)
        th_ctx[:len_ctx] = torch.from_numpy(ctx)[:len_ctx].long()
        th_tgt = torch.full((self.max_target_length + 1,), -100, dtype=torch.long)
        # th_tgt[0] = 1
        # th_tgt[1:1+len_tgt] = torch.from_numpy(tgt)[:len_tgt].long()
        th_tgt[:len_tgt] = torch.from_numpy(tgt)[:len_tgt].long() # TODO
        return th_ctx, th_tgt, len_ctx, len_tgt

    def __getitem__(self, index):
        ctx = self.ctx[index]
        tgt = self.tgt[index]

        if isinstance(index, int):
            th_ctx, th_tgt, len_ctx, len_tgt = self.__get_item_data(ctx, tgt)
            if th_ctx is None:
                return None
            return {
                "ctx": th_ctx,
                "tgt": th_tgt,
                "len_ctx": len_ctx,
                "len_tgt": len_tgt
            }
        else:
            res = {"ctx": [], "tgt": [], "len_ctx": [], "len_tgt":[]}
            for _ctx, _tgt in zip(ctx, tgt):
                _th_ctx, _th_tgt, _len_ctx, _len_tgt = self.__get_item_data(_ctx, _tgt)
                if _th_ctx is None:
                    continue
                res["ctx"].append(_th_ctx)
                res["tgt"].append(_th_tgt)
                res["len_ctx"].append(_len_ctx)
                res["len_tgt"].append(_len_tgt)
            return {
                "ctx": torch.stack(res["ctx"]), 
                "tgt": torch.stack(res["tgt"]),
                "len_ctx": torch.LongTensor(res["len_ctx"]),
                "len_tgt": torch.LongTensor(res["len_tgt"])
            }

