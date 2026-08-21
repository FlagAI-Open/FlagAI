import json
import os
import sys
import shutil
import numpy as np
import datetime
import webdataset as wds
from multiprocessing import Process
from PIL import Image

def make_wds_shards(pattern, num_shards, num_workers, src, label, map_func, **kwargs):
    src_per_shards = [src[i::num_shards] for i in range(num_shards)]
    label_per_shards = [label[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                src_per_shards[i::num_workers],
                label_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, src, label, map_func, kwargs):
    for shard_id, src, label in zip(shard_ids, src, label):
        write_samples_into_single_shard(pattern, shard_id, src, label, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, srcs, labels, map_func, kwargs):
    fname = pattern % shard_id
    # print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    sink = wds.TarWriter(fname, **kwargs)
    
    for content in map_func(srcs, labels):
        sink.write(content)
    sink.close()
    # print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}")


src_dir = "./data/train.src"
tgt_dir = "./data/train.tgt"

def read_file():
    src = []
    tgt = []

    with open(src_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            src.append(line.strip('\n').lower())

    with open(tgt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n').lower())

    return src, tgt

    

if __name__ == "__main__":
    num_shards = 231350
    num_workers = 10
    src, tgt = read_file()
    num_shards = len(tgt)
    output_path = './webdataset'


    
    def sampler(src, tgt):
        """
        keys will automatically be '0.png' and '0.txt' because the image name is xxxx.0.png/
        """
        
        for i in range(len(src)):
            sample = {
                "__key__": str(i),
                "src": src[i],
                "tgt": tgt[i]
            }
            yield sample

    make_wds_shards(
        pattern=f"{output_path}/%06d.tar",
        num_shards=num_shards, # 设置分片数量
        num_workers=num_workers, # 设置创建wds数据集的进程数
        src=src,
        label=tgt,
        map_func=sampler,
    )