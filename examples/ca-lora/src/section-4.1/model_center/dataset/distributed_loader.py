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
import torch.utils.data as data
import bmtrain as bmt

class DistributedDataLoader:
    def __init__(self, dataset, shuffle=False, **kwargs):
        self.sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle, rank=bmt.rank(), num_replicas=bmt.world_size())
        self.loader = data.DataLoader(dataset, shuffle=False, sampler=self.sampler, **kwargs)
        self.epoch = 0
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
        self.sampler.set_epoch(self.epoch)
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
    