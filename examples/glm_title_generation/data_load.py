import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    print(shards_list)
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def collate_fn(batch):
        
        return {
            "src": batch[0],
            "tgt": batch[1]
        }


def decode_samples(samples):
    for sample in samples:

        sample['src'] = sample['src'].decode('utf-8')
        sample['tgt'] = sample['tgt'].decode('utf-8')
        # print(sample)
        yield sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(data_path, is_train, 
                    batch_size=32, dataset_resampled=False,
                    epoch=0, floor=False, tokenizer=None):
    input_shards = data_path

    assert input_shards is not None
    resampled = dataset_resampled

    num_samples = 10000
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    pdb.set_trace()
    pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    
    # at this point we have an iterator over all the shards
    if is_train:
       
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        decode_samples,
        # wds.map_dict(src=lambda src: tokenizer(src)[0]),
        wds.to_tuple("src", "tgt"),
        wds.batched(batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = 4
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)

    
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



if __name__ == "__main__":
    import os
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from flagai.auto_model.auto_loader import AutoLoader
    from flagai.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trainer = Trainer(
    #     env_type="bmtrain",
    #     experiment_name="GLM_seq2seq",
    #     batch_size=1,
    #     gradient_accumulation_steps=1,
    #     lr=2e-4,
    #     weight_decay=1e-5,
    #     epochs=10,
    #     num_gpus = 2,
    #     log_interval=10,
    #     eval_interval=40,
    #     load_dir=None,
    #     pytorch_device=device,
    #     # save_dir="checkpoints_glm_title_generation",
    #     save_interval=40,
    #     num_checkpoints=1,
    #     hostfile='./hostfile',
    #     training_script=__file__,
    #     deepspeed_config='./deepspeed.json'
    # )


    auto_loader = AutoLoader("lm",
                         model_name="GLM-large-ch",
                         model_dir="./state_dict/")
    model = auto_loader.get_model()
    tokenizer = auto_loader.get_tokenizer()
    pdb.set_trace()

    ds = get_wds_dataset("./webdataset/000000.tar", 
                        tokenizer=tokenizer, 
                        is_train=True, 
                        epoch=1)

    for i in ds.dataloader :
        print(len(i['src']))
        # print(samples.shape)
        # print(target.shape)
        break 