import torch
import torch.utils.data as data
from .indexed import MMapIndexedDataset
import random
import numpy as np

class CPM3_Dataset_Merge(data.Dataset):
    def __init__(self, ctx : MMapIndexedDataset, info: MMapIndexedDataset, 
                 max_length = 2048, prompt_length = 64):
        self.ctx = ctx
        self.info = info
        self.max_length = max_length
        self.prompt_length = prompt_length

    def __len__(self):
        return len(self.ctx)
    
    def __get_item_data(self, ctx, info):
        
        # 超长就跳过该条数据
        if ctx.shape[0] > self.max_length - self.prompt_length:
            return None, None, None, None, None, None, None

        task = info[0]
        len_ctx = min(ctx.shape[0], self.max_length - self.prompt_length)
        inp = np.arange((self.prompt_length+len_ctx), dtype = np.int64) + self.prompt_length * task
        inp[self.prompt_length:] = ctx[:len_ctx]
        len_inp = len(inp)

        info = info[1:] + self.prompt_length
        context_inp = np.full(len_inp, True)
        # 保证附加的eos一定能看见
        for i in range(1, len(info)-1, 2):
            context_inp[info[i]:info[i+1]] = False
        
        tgt = np.full((len_inp), -100, dtype = np.int64)
        tgt[:-1] = np.where(
            context_inp[1:],
            -100,
            inp[1:]
        )

        position_inp = np.arange((len_inp), dtype = np.float32) / self.prompt_length
        segment_inp = np.zeros((len_inp), dtype = np.int64)
        task_inp = np.full((len_inp), task, dtype = np.int64)

        if task == 0:
            arr = [(2, info[0]), (1, 0), (1, info[-1])]
        else:
            arr = [(2, info[0]), (2+task, info[1]), (1, info[-1])]
        
        last = self.prompt_length
        for (typ, end) in arr:
            if end > last:
                segment_inp[last:end] = typ
                position_inp[last:end] = np.arange(end-last) / (end-last)
                last = end
        assert last == len_inp
        # print("inp:\n", inp)
        # print("tgt:\n", tgt)
        # print("len_input:\n", len_inp)
        # print("context_inp:\n", context_inp)
        # print("position_inp:\n", position_inp)
        # print("segment_inp:\n", segment_inp)
        return inp, tgt, len_inp, context_inp, position_inp, segment_inp, task_inp

    def __getitem__(self, index):
        ctx = self.ctx[index]
        info = self.info[index]
        th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx = \
                self.__get_item_data(ctx, info)
        return th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx
