# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT style dataset."""

import os
import time

import numpy as np
import torch

from megatron import mpu, print_rank_0
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.gpt_dataset import _build_shuffle_idx, _build_doc_idx, _num_epochs, _num_tokens, get_indexed_dataset_, _build_sample_idx

class GPTDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed):

        self.name = name
        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name, data_prefix, documents, self.indexed_dataset.sizes,
            num_samples, seq_length, seed)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = idx % len(self)
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)

        return {'input_ids': np.array(sample, dtype=np.int64)}


def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if True:
        if (not os.path.isfile(doc_idx_filename)) or \
           (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                      'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                    (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                print_rank_0(f"num_samples_per_epoch={num_samples_per_epoch}, last_epoch_num_samples={last_epoch_num_samples}")
                #assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                #    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                       int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger '\
                             'than 80% of number of samples per epoch ({}), '\
                             'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            # from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            #assert sizes.dtype == np.int32
            sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx

class BlendableDataset(torch.utils.data.Dataset):

    def __init__(self, datasets, weights, max_num_samples=None):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)
        print('BlendableDataset init size', self.size)
        if max_num_samples is not None:
            self.size = max_num_samples
            print('BlendableDataset actual size', self.size)

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        from megatron.data import helpers
        helpers.build_blending_indices(self.dataset_index,
                                       self.dataset_sample_index,
                                       weights, num_datasets, self.size,
                                       True)
        print_rank_0('> elapsed time for building blendable dataset indices: '
                     '{:.2f} (sec)'.format(time.time() - start_time))


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx]

def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            print_rank_0(f"Started build_dataset {data_prefix}")
            dataset = GPTDataset(name, data_prefix,
                                  documents, indexed_dataset,
                                  train_valid_test_num_samples[index],
                                  seq_length, seed)
            print_rank_0(f"Ended build_dataset {data_prefix}")
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)

def _build_train_valid_test_weighted_datasets(
    data_prefix, data_impl, splits_string,
    train_valid_test_num_samples,
    seq_length, seed, skip_warmup,
    train_max_num_samples=None):
    """Build train, valid, and test datasets."""

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output
    #print('prefixes', prefixes)
    #total = 0
    #for xy in datasets_train_valid_test_num_samples:
    #    total += xy[0]
    #print('datasets_train_valid_test_num_samples', datasets_train_valid_test_num_samples)
    #print('datasets_train_valid_test_num_samples total', total)

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i], data_impl, splits_string,
            datasets_train_valid_test_num_samples[i],
            seq_length, seed, skip_warmup)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights, max_num_samples=train_max_num_samples)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)

if __name__ == '__main__':
    ### 需要根据数据集情况填写
    ### documents_stat.py
    ### 样本量和epochs提前考虑,这里统一做打散

    ### gpt2
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpt2/merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [41313229, 4132, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = False

    ### gpm
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [343969381, 344314, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = False

    ### debug
    data_prefix = '2000_text_document/merged_text_document'
    data_impl = 'lazy'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [10, 1, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = False

    ### debug
    data_prefix = '00_text_document/00_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [1, 1, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = False

    ### gpm part
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/part_merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [99136540, 99236, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = False

    ### gpm 10
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/10_merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [29375962, 29406, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = False

    ### gpm 20
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/20_merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [70166341, 70237, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = False

    ### gpm 12
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/12_merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [33605368, 33606, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = False

    ### gpm debug
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/debug_merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [29375962, 29406, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = False

    ### gpm
    data_prefix = '/share/project/ldwang/data/indexed_dataset/gpm/merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [344379254, 34441, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    ### wikitext
    data_prefix = '/home/ldwang/Downloads/pile/wikitext_text_document'
    data_impl = 'mmap'
    splits_string = '10000,0,0'
    train_valid_test_num_samples = [2891, 0, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = True

    ### lambada
    data_prefix = '/home/ldwang/Downloads/pile/lambada_text_document'
    data_impl = 'mmap'
    splits_string = '10000,0,0'
    train_valid_test_num_samples = [5153, 0, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = True

    ### webtext
    data_prefix = '/home/ldwang/Downloads/pile/webtext_text_document'
    data_impl = 'mmap'
    splits_string = '10000,0,0'
    train_valid_test_num_samples = [250000, 0, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = True

    ### OpenWebText2
    data_prefix = '/share/project/ldwang/data/indexed_dataset/merged/OpenWebText2_merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [26944801, 2695, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = True

    ### dedup_wudao_5pct
    data_prefix = '/share/project/ldwang/data/indexed_dataset/merged/dedup_wudao_5pct_merged_text_document'
    data_impl = 'mmap'
    splits_string = '9999,1,0'
    train_valid_test_num_samples = [10500000, 1050, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    '''
    train_dataset, valid_dataset, test_dataset = _build_train_valid_test_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup)
    print(len(train_dataset))
    print(type(train_dataset))
    print(train_dataset[0])
    '''

    ## weight01, prefix01, weight02, prefix02
    data_prefix = [
        200,
        '/share/project/ldwang/data/indexed_dataset/merged/OpenWebText2_merged_text_document',
        100,
        '/share/project/ldwang/data/indexed_dataset/merged/dedup_wudao_5pct_merged_text_document',
    ]
    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    splits_string = '9999,1'
    ## rebuilding if no npy files for train_valid_test_num_samples config
    train_valid_test_num_samples = [26944801, 2695]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    print_rank_0(f"Started _build_train_valid_test_weighted_datasets")
    train_dataset, valid_dataset, test_dataset = _build_train_valid_test_weighted_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup,
        train_max_num_samples)
    #print(len(train_dataset))
    #print(len(valid_dataset))
    #print(train_dataset[37735074])
    #print(type(train_dataset))
    #print(train_dataset[0])
    '''
    '''

    '''
    ### debug
    data_prefix = '/home/ldwang/Downloads/pile/debug_text_document'
    data_impl = 'mmap'
    splits_string = '100,0'
    train_valid_test_num_samples = [3, 0]
    seq_length = 1024
    seq_length = 256
    seq_length = 380
    seed = 2023
    skip_warmup = True

    ### wikitext_concat
    data_prefix = '/home/ldwang/Downloads/pile/wikitext_concat_text_document'
    data_impl = 'mmap'
    splits_string = '10000,0'
    train_valid_test_num_samples = [1, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = True

    ### lambada_concat
    data_prefix = '/home/ldwang/Downloads/pile/lambada_concat_text_document'
    data_impl = 'mmap'
    splits_string = '10000,0'
    train_valid_test_num_samples = [1, 0]
    seq_length = 1024
    seed = 2023
    skip_warmup = True

    train_dataset, valid_dataset, _ = _build_train_valid_test_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup)
    print('len(train_dataset)', len(train_dataset))
    for batch in train_dataset:
        print('len batch', len(batch['input_ids']))
        print('batch', batch['input_ids'])

    '''

    '''
    import sys
    assert len(sys.argv)>1

    data_prefix = sys.argv[1]
    data_impl = 'mmap'
    # Indexed dataset.
    dataset = get_indexed_dataset_(
        data_prefix,
        data_impl=data_impl,
        skip_warmup=True)
    total_num_of_documents = dataset.sizes.shape[0]

    splits_string = '9999,1,0'
    train_num_samples = int(total_num_of_documents*0.9999)
    valid_num_samples = total_num_of_documents-train_num_samples
    train_valid_test_num_samples = [train_num_samples, valid_num_samples, 0]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    train_dataset, valid_dataset, _ = _build_train_valid_test_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup)
    print('len(train_dataset)', len(train_dataset))
    print('last(train_dataset)', train_dataset[len(train_dataset)-1].keys())
    '''

