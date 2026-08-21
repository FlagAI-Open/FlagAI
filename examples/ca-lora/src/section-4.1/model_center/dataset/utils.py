import os
from typing import List
from .distributed_dataset import SimpleDataset, build_dataset, _random_string, _DEFAULT_BLOCK_SIZE, FileInfo
import random
import pickle

try:
    from tqdm import tqdm
    support_tqdm = True
except ModuleNotFoundError:
    support_tqdm = False


_DEFAULT_SHUFFLE_BUCKET_SIZE = 1 << 30
def shuffle_dataset(path_src : str, path_tgt : str, block_size : int = _DEFAULT_BLOCK_SIZE, bucket_size : int = _DEFAULT_SHUFFLE_BUCKET_SIZE, progress_bar = False):
    """ Shuffle one distributed datataset, write results to another dataset.

    Args:
        path_str (str): path to source dataset
        path_tgt (str): path to write results
        block_size (int): dataset block size (default: 16MB)
        bucket_size (int): shuffle algorithm bucket size (default: 1GB)
        progress_bar (bool): show progress bar
    
    Example:
        >>> shuffle_dataset("/path/to/source", "/path/to/output")
    """

    if progress_bar and not support_tqdm:
        raise RuntimeError("Requires `tqdm` to enable progress bar.")
    
    ds = SimpleDataset(path_src, block_size=block_size)
    num_buckets = (ds.nbytes + bucket_size - 1) // bucket_size

    tmp_files = [
        os.path.join(path_src, ".tmp.%s" % _random_string()) for _ in range(num_buckets)
    ]

    try:
        # Step 1: write to bucket randomly
        f_tmp = [ open(fname, "wb") for fname in tmp_files ]
        try:
            iterator = ds
            if progress_bar:
                iterator = tqdm(ds, desc="Shuffle step 1/2")
            for data in iterator:
                bucket_id = int(random.random() * num_buckets)
                pickle.dump(data, f_tmp[bucket_id]) # write into a random bucket
        finally:
            # close all files
            for fp in f_tmp:
                if not fp.closed:
                    fp.close()
        f_tmp = []

        # Step 2: shuffle inside bucket
        with build_dataset(path_tgt, "%s.shuffle" % _random_string()) as writer:
            iterator = tmp_files
            if progress_bar:
                iterator = tqdm(tmp_files, desc="Shuffle step 2/2")

            for fname in iterator:
                fp = open(fname, "rb")
                data_in_bucket = []
                while True:
                    try:
                        data_in_bucket.append(fp.read())
                    except EOFError:
                        break
                random.shuffle(data_in_bucket)
                for data in data_in_bucket:
                    writer.write(data)
                fp.close()
                os.unlink(fname)
    finally:
        # cleanup
        for fname in tmp_files:
            if os.path.exists(fname):
                os.unlink(fname)

def compact_dataset(path : str):
    """ Compact the dataset, removes blocks which the files were deleted.

    **Note** This may affect the existing dataset state dict.

    Args:
        path (str): path to dataset

    Example:
        >>> compact_dataset("/path/to/dataset")
    
    """

    meta_path = os.path.join(path, "meta.bin")

    info : List[FileInfo] = []
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            info = pickle.load(f)
    else:
        raise ValueError("Dataset not exists")
    
    nw_info = []
    curr_block = 0
    for v in info:
        if not os.path.exists(v.file_name):
            # file is deleted
            pass
        else:
            num_file_block = v.block_end - v.block_begin
            nw_info.append(FileInfo(
                v.file_name,
                curr_block,
                curr_block + num_file_block,
                v.nbytes,
                v.nlines,
                v.mask
            ))
            curr_block += num_file_block
    
    random_fname = os.path.join(path, ".meta.bin.%s" % _random_string())
    with open(random_fname, "wb") as f:
        pickle.dump(nw_info, f)
    os.rename(random_fname, meta_path)

def mask_dataset(path : str, dbname : str, mask : bool = True):
    """ Mask one file in dataset. Blocks in masked datasets won't be read later.

    Args:
        path (str): path to dataset
        dbname (str): file name in this dataset which you want to mask
        mask (bool): True for mask, False for unmask

    Example:
        >>> mask_dataset("/path/to/dataset", "data_part_1", mask=True)
    
    """

    meta_path = os.path.join(path, "meta.bin")

    info : List[FileInfo] = []
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            info = pickle.load(f)
    else:
        raise ValueError("Dataset not exists")
    
    for v in info:
        if v.file_name == dbname:
            v.mask = mask
    
    random_fname = os.path.join(path, ".meta.bin.%s" % _random_string())
    with open(random_fname, "wb") as f:
        pickle.dump(info, f)
    os.rename(random_fname, meta_path)
