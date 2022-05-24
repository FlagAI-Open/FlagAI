import re
import torch
import argparse
import time
from tqdm import tqdm
import random
import numpy as np
from torch import cuda
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from flagai.model.t5_model import T5UERModel
from flagai.data.tokenizer.t5.t5_tokenizer import T5JiebaTokenizer, load_chinese_base_vocab
import torch.nn.functional as F
import json
import html
import pandas as pd
from flagai.trainer import Trainer


def train_data_process_kgclue(file):
    source, target = [], []
    f = pd.read_csv(file)
    for question, entity in zip(f['question'], f['entity']):
        source.append(question)
        target.append(entity)
    return (source, target)


class SeqDataset(Dataset):
    """
    Dataset function
    """

    def __init__(self, sents_src, sents_tgt):
        super(SeqDataset, self).__init__()
        
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in word2idx.items()}

    def __getitem__(self, i):
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids_src, _ = tokenizer.encode(src, max_length=256)
        token_ids_tgt, _ = tokenizer.encode(tgt, max_length=256)
        output = {
            "token_ids_src": token_ids_src,
            "token_ids_tgt": token_ids_tgt,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch):
    """
    dynamic padding， 
    """

    def padding(indice, max_length, pad_idx=0):
        """
        paddiing function
        """
        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice)

    token_ids_src = [data["token_ids_src"] for data in batch]
    max_length_src = max([len(t) for t in token_ids_src])
    token_ids_tgt = [data["token_ids_tgt"] for data in batch]
    max_length_tgt = max([len(t) for t in token_ids_tgt])

    token_ids_padded = padding(token_ids_src, max_length_src)
    target_ids_padded = padding(token_ids_tgt, max_length_tgt)
    labels_ids = target_ids_padded.clone()
    labels_ids[labels_ids == 0] = -100
    target_ids_padded = target_ids_padded[:, :-1].contiguous()
    labels_ids = labels_ids[:, 1:].contiguous()

    return {
        'input_ids': token_ids_padded,
        'decoder_input_ids': target_ids_padded,
        'labels': labels_ids
    }


if __name__ == '__main__':
    data_name = 'kgclue'
    train_path = './data/train.csv'
    test_path = './data/dev.csv'
    model_name = 'T5'
    vocab_path = '/mnt/T5_JIEBA/vocab.txt'
    model_path = '/mnt/T5_JIEBA/pytorch_model.bin'
    logger.add('log/log_' + data_name + '_' + model_name +
               '_{time}.log')  # 
    torch.cuda.empty_cache()  
    # load model
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenizer = T5JiebaTokenizer(token_dict=word2idx)
    model = T5UERModel(word2idx)
    model.load_pretrain_params(model_path)
    # put model to GPU:rank, one process corresponding to one gpu. this is different model.cuda() in nn.DataParallel()

    T5Trainer = Trainer(
        env_type="pytorch",
        experiment_name="roberta_ner",
        batch_size=64,
        lr=2e-4,
        weight_decay=1e-3,
        epochs=10,
        load_dir=None,
        save_dir="kgclue",
        save_epoch=1,
        eval_interval=False,
        seed=0,
    )

    train_data = train_data_process_kgclue(train_path)
    ttest_data = train_data_process_kgclue(test_path)
    trainset = SeqDataset(train_data[0], train_data[1])
    train_params = {
        "batch_size": T5Trainer.batch_size,
        "shuffle": True,
        "num_workers": 4,
        "collate_fn": collate_fn
    }
    training_loader = DataLoader(trainset, **train_params)
    iter_training_loader = iter(training_loader)
    data = next(iter_training_loader)
    T5Trainer.train(model, train_dataset=training_loader)
