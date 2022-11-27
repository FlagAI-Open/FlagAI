import os
import struct
import shutil

from itertools import accumulate

import numpy as np
import torch

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


class DistributedMMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'
        def __init__(self, path):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, name, rank_number, rank_total, cache = None):
        
        super().__init__()

        self._path = path
        self._name = name
        self._state = 0
        if cache is not None:
            self._cache = cache
            os.makedirs(self._cache, exist_ok=True)
        else:
            self._cache = None
        self._rank_total = rank_total
        self._rank_number = rank_number
        self._index = None
        self._bin_buffer = None
        self._bin_buffer_mmap = None
        self.history = {-1:0}

        self._do_init(self._path, self._name, self._cache, self._state)

    def __getstate__(self):
        return self._path + self._name + "_%d"%(self._state)

    def __setstate__(self, state):
        self._state = state
        self._do_init(self._path, self._name, self._cache, self._state)

    def _do_init(self, path, name, cache, state):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

        self._state = state

        source_file = path + name
        self._index = self.Index(index_file_path(source_file))
        self._bin_buffer_mmap = np.memmap(data_file_path(source_file), mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        self.history[state] = self.history[state - 1] + len(self._index) // self._rank_total
        self.start = (len(self._index) // self._rank_total) * self._rank_number
        print (self.history[state],"======================", self.start, self.history[self._state - 1])

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

    def _next_file(self):
        self._state += 1
        self._do_init(self._path, self._name, self._cache, self._state)
    
    def __relative_idx(self, idx):
        res = self.start + idx - self.history[self._state - 1]
        return res

    def __slice_item(self, start, stop):
        ptr = self._index._pointers[self.__relative_idx(start)]
        sizes = self._index._sizes[self.__relative_idx(start):self.__relative_idx(stop)]
        offsets = list(accumulate(sizes))
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=sum(sizes), offset=ptr)
        return np.split(np_array, offsets[:-1])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            while idx >= self.history[self._state]:
                self._next_file()
            ptr, size = self._index[self.__relative_idx(idx)]
            return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(2147483647)
            assert step == 1 or step is None, "Slices into indexed_dataset must be contiguous"
            if stop >= self.history[self._state]:
                res_1 = self.__slice_item(start, self.history[self._state]) 
                self._next_file()
                res_2 = self.__slice_item(self.history[self._state - 1], stop)
                return res_1 + res_2
            else:
                return self.__slice_item(start, stop)

    def __len__(self):
        return self.history[self._state]

    @property
    def sizes(self):
        return self._index.sizes
        
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )
