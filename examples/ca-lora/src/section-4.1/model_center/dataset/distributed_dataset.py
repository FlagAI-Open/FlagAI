import io
import os
import pickle
from typing import List
import torch
import bisect
import bmtrain as bmt

import random
import string

def _random_string():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

class FileInfo:
    def __init__(
            self,
            file_name : str,
            block_begin : int,
            block_end : int,
            nbytes : int,
            nlines : int,
            mask : bool = False
        ) -> None:
        self.file_name = file_name
        self.block_begin = block_begin
        self.block_end = block_end
        self.nbytes = nbytes
        self.nlines = nlines
        self.mask = mask
    
    @classmethod
    def _load_from_state(cls, version, data):
        if version == 1:
            file_name, block_begin, block_end, nbytes, nlines, mask = data
            return cls(file_name, block_begin, block_end, nbytes, nlines, mask)
        else:
            raise RuntimeError("Unsupported version %d" % version)

    def __reduce__(self):
        return (FileInfo._load_from_state, (1, (self.file_name, self.block_begin, self.block_end, self.nbytes, self.nlines, self.mask)))


_MASK_VALUE = 0x7fffffff
_DEFAULT_BLOCK_SIZE = 16 << 20

class DistributedDataset:
    """ Open dataset in readonly mode.
    
    `DistributeDataset` is used to read datasets in a distributed manner.
    Data in this dataset will be distributed evenly in blocks to each worker in the `distributed communicator`.
    
    **Note** When all data has been read, reading dataset again will revert back to the first data.

    Args:
        path (str): Path to dataset.
        rank (int): Rank in distributed communicator. See: bmtrain.rank()
        world_size (int): Total workers in distributed communicator. See: bmtrain.world_size()
        block_size (int): Size of each block in bytes. All files in the same dataset should have the same block size. Default: 16MB
    
    Example:
        >>> dataset = DistributedDataset("/path/to/dataset")
        >>> for i in range(10):
        >>>     dataset.read()
    """
    def __init__(self, path : str, rank : int = 0, world_size : int = 1, block_size = _DEFAULT_BLOCK_SIZE) -> None:
        # config
        self._path = path
        self._rank = rank
        self._world_size = world_size
        self._block_size = block_size

        # dataset meta
        self._block_states = torch.tensor([], dtype=torch.int)
        self._file_info : List[FileInfo] = []
        self._file_ends : List[int] = []
        self._total_blocks = 0
        self._nbytes = 0
        self._nlines = 0

        # states
        self._curr_block = None
        self._fp = None

        # cache
        self._last_mod_time = 0
        self._curr_fname = None

        self._update_states()

    def _update_states(self, fast_skip : bool = True):
        meta_path = os.path.join(self._path, "meta.bin")

        mod_time = os.stat(meta_path).st_mtime
        if self._last_mod_time < mod_time:
            # file changed
            pass
        else:
            if fast_skip:
                return

        info : List[FileInfo] = []
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                info = pickle.load(f)
        
        old_len = len(self._file_info)
        if old_len > len(info):
            raise RuntimeError("Dataset meta file: changed unexpectly")
        for i in range(old_len):
            if self._file_info[i].file_name != info[i].file_name:
                raise RuntimeError("Dataset meta file: changed unexpectly")
            if self._file_info[i].block_begin != info[i].block_begin:
                raise RuntimeError("Dataset meta file: changed unexpectly")
            if self._file_info[i].block_end != info[i].block_end:
                raise RuntimeError("Dataset meta file: changed unexpectly")

        if info[0].block_begin != 0:
            raise RuntimeError("Dataset meta file: block error (0)")
        for i in range(len(info) - 1):
            if info[i].block_end != info[i + 1].block_begin:
                raise RuntimeError("Dataset meta file: block error (%d)" % (i + 1))

        if old_len == len(info) and fast_skip:
            # fast skip
            return

        if len(info) > 0:
            total_blocks = info[-1].block_end
            self._nbytes = 0
            self._nlines = 0
            for v in info:
                self._nbytes += v.nbytes
                self._nlines += v.nlines
        else:
            total_blocks = 0
            self._nbytes = 0
            self._nlines = 0
        
        if total_blocks > 0:
            masks = torch.full((total_blocks,), _MASK_VALUE, dtype=torch.int, device="cpu", requires_grad=False)
            masks[self._rank::self._world_size] = 0
            for v in info:
                if v.mask or (not os.path.exists(self._get_file_path(v.file_name))):
                    masks[v.block_begin:v.block_end] = _MASK_VALUE
            new_block_states = torch.zeros(total_blocks, dtype=torch.int, device="cpu", requires_grad=False)
            new_block_states[:self._block_states.size(0)] = self._block_states
            new_block_states = torch.maximum(new_block_states, masks)

            self._block_states = new_block_states
            
            self._file_ends = []
            for v in info:
                self._file_ends.append(v.block_end)
        else:
            self._block_states = torch.tensor([], dtype=torch.int, device="cpu", requires_grad=False)
            self._file_ends = []
        self._total_blocks = total_blocks
        self._file_info = info

        assert len(self._file_ends) == len(self._file_info)

    def _mask_file(self, f : FileInfo):
        masks = torch.full((self._total_blocks,), 0, dtype=torch.int, device="cpu", requires_grad=False)
        masks[f.block_begin:f.block_end] = _MASK_VALUE
        self._block_states = torch.maximum(self._block_states, masks)

    def _get_block_file(self, block_id : int):
        # find block in which file
        file_idx = bisect.bisect_right(self._file_ends, block_id)
        return self._file_info[file_idx]

    def _get_next_block(self):
        self._update_states()
        if self._block_states.size(0) == 0:
            raise RuntimeError("Empty dataset")
        mn_block = self._block_states.argmin().item()
        if self._block_states[mn_block].item() == _MASK_VALUE:
            raise RuntimeError("Empty dataset")
        self._block_states[mn_block] += 1
        return mn_block
    
    def state_dict(self):
        """ Returns a state dict representing the read states of the dataset.

        Example:
            >>> state = dataset.state_dict()
            >>> dataset.load_state_dict(state)
        """
        self._update_states()
        states = torch.where(
            self._block_states == _MASK_VALUE,
            torch.zeros(self._total_blocks, dtype=torch.int, device="cpu", requires_grad=False),
            self._block_states
        )

        if self._curr_block is not None:
            curr_block = self._curr_block
            curr_f = self._get_block_file(curr_block)
            inblock_offset = self._fp.tell() - (curr_block - curr_f.block_begin) * self._block_size
        else:
            curr_block = -1
            inblock_offset = 0

        with torch.no_grad():
            if self._world_size > 1:
                gpu_states = states.cuda()
                gpu_block = torch.tensor([ curr_block, inblock_offset ], dtype=torch.long).cuda()
                global_states = bmt.distributed.all_reduce(gpu_states, op="sum").cpu()
                global_block = bmt.distributed.all_gather(gpu_block).cpu()
                return {
                    "states": global_states,
                    "block": global_block
                }
            else:
                return {
                    "states": states,
                    "block": torch.tensor([[curr_block, inblock_offset]], dtype=torch.long, device="cpu")
                }
    
    def load_state_dict(self, state, strict : bool = True):
        """ Load dataset state.

        Args:
            state (dict): dataset state dict.
            strict (bool): If `strict` is True, world size needs to be the same as when exported.

        Example:
            >>> state = dataset.state_dict()
            >>> 
        """

        self._block_states = state["states"]
        self._update_states(False)

        if state["block"].size(0) != self._world_size:
            if strict:
                raise ValueError("world_size changed (%d -> %d)" % (state["block"].size(0), self._world_size))
            else:
                self._curr_block = None
                self._fp = None
                self._curr_fname = None
        else:
            curr_block = state["block"][self._rank][0].item()
            inblock_offset = state["block"][self._rank][1].item()

            if curr_block == -1:
                self._curr_block = None
            else:
                self._curr_block = curr_block
                f_info = self._get_block_file(self._curr_block)
                self._open_file(f_info.file_name, (self._curr_block - f_info.block_begin) * self._block_size + inblock_offset)
        # end

    def _get_file_path(self, fname):
        return os.path.join(self._path, fname)

    def _open_file(self, fname, offset):
        if self._curr_fname != fname:
            if self._fp is not None:
                self._fp.close()
                self._curr_fname = None
            self._fp = open(self._get_file_path(fname), "rb")
            self._curr_fname = fname
        self._fp.seek(offset, io.SEEK_SET)    # move to block

    def read(self):
        """ Read a piece of data from dataset.

        Workers in different ranks will read different data.
        """
        if self._curr_block is None:
            next_block_id = self._get_next_block()
            f_info = self._get_block_file(next_block_id)
            try:
                self._open_file(f_info.file_name, (next_block_id - f_info.block_begin) * self._block_size)
                self._curr_block = next_block_id
            except FileNotFoundError:
                self._mask_file(f_info)
                return self.read()  # read again

        MAGIC = self._fp.read(1)
        if MAGIC == b"\x1F":
            # correct
            return pickle.load(self._fp)
        elif MAGIC == b"\x00":
            # end of block
            self._curr_block = None
            return self.read()  # read next block
        else:
            raise ValueError("Invalid magic header")
    
    @property
    def nbytes(self):
        return self._nbytes


class SimpleDataset(DistributedDataset):
    def __init__(self, path: str, block_size=_DEFAULT_BLOCK_SIZE) -> None:
        super().__init__(path, 0, 1, block_size)
    
    def _get_next_block(self):
        self._update_states()
        if self._block_states.size(0) == 0:
            raise RuntimeError("Empty dataset")
        mn_block = self._block_states.argmin().item()
        if self._block_states[mn_block].item() >= 1:
            raise EOFError("no more data")
        self._block_states[mn_block] += 1
        return mn_block
    
    def __iter__(self):
        while True:
            try:
                data = self.read()
            except EOFError:
                break
            yield data
    
    def __len__(self):
        return self._nlines

class DatasetWriter:
    def __init__(self, fname, block_size):
        self._fname = fname
        self._block_size = block_size
        self._fp = open(self._fname, "wb")
        self._inblock_offset = 0

        self._nbytes = 0
        self._nlines = 0
        self._nblocks = 1
    
    def write(self, data):
        """ Write a piece of data into dataset.

        Args:
            data (Any): Serialization will be done using pickle.

        Example:
            >>> writer.write( "anything you want" )

        """
        byte_data = pickle.dumps(data)
        if self._inblock_offset + 2 + len(byte_data) > self._block_size:
            self._fp.write(b'\x00' * (self._block_size - self._inblock_offset)) # fill the remaining space with 0
            self._inblock_offset = 0
            self._nblocks += 1
            # we go to the next block
        if self._inblock_offset + 2 + len(byte_data) > self._block_size:
            raise ValueError("data is larger than block size")
        
        self._nbytes += len(byte_data)
        self._nlines += 1

        self._inblock_offset += 1 + len(byte_data)
        self._fp.write(b"\x1F")
        self._fp.write(byte_data)

    @property
    def nbytes(self):
        return self._nbytes
    
    @property
    def nblocks(self):
        return self._nblocks
    
    @property
    def nlines(self):
        return self._nlines

    def close(self):
        if not self._fp.closed:
            self._fp.write(b"\x00" * (self._block_size - self._inblock_offset)) 
            self._fp.close()


class DatasetBuilder:
    def __init__(self, path : str, dbname : str, block_size = _DEFAULT_BLOCK_SIZE) -> None:
        self._block_size = block_size
        self._path = path
        self._dbname = dbname

        if not os.path.exists(self._path):
            os.makedirs(self._path)

        meta_path = os.path.join(self._path, "meta.bin")

        info : List[FileInfo] = []
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                info = pickle.load(f)
        
        for v in info:
            if v.file_name == dbname:
                raise ValueError("Dataset name exists")
            
        self._db_path = os.path.join(self._path, self._dbname)
        if os.path.exists(self._db_path):
            raise ValueError("File exists `%s`" % self._db_path)
    
    def __enter__(self):
        self._writer = DatasetWriter(self._db_path, self._block_size)
        return self._writer
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._writer.close()
        if exc_type is not None:
            print("Error while writing file")
            if os.path.exists(self._db_path):
                os.unlink(self._db_path)
        else:
            meta_path = os.path.join(self._path, "meta.bin")
            info : List[FileInfo] = []
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    info = pickle.load(f)
            last_block = 0
            if len(info) > 0:
                last_block = info[-1].block_end
            info.append(FileInfo(
                self._dbname,
                last_block,
                last_block + self._writer.nblocks,
                self._writer.nbytes,
                self._writer.nlines,
                False
            ))

            # atomic write to meta file
            random_fname = os.path.join(self._path, ".meta.bin.%s" % _random_string())
            with open(random_fname, "wb") as f:
                pickle.dump(info, f)
            os.rename(random_fname, meta_path)
        
        self._writer = None

def build_dataset(path : str, dbname : str, block_size : int = _DEFAULT_BLOCK_SIZE):
    """ Open the dataset in write mode and returns a writer.

    Args:
        path (str): Path to dataset.
        dbname (str): The name of the file to which the data will be written. The `dbname` needs to be unique in this `dataset`.
        block_size (int): Size of each block in bytes. All files in the same dataset should have the same block size. Default: 16MB
    
    Example:
        >>> with build_dataset("/path/to/dataset", "data_part_1") as writer:
        >>>     for i in range(10):
        >>>         writer.write( { "anything you want" } )
    """
    return DatasetBuilder(path, dbname, block_size)
