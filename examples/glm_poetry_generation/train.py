# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer

cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = cur_dir + '/data/src.txt'
tgt_dir = cur_dir + '/data/tgt.txt'
model_dir = "./state_dict/"  # ./state_dict/roberta/  # 模型位置


def read_file():
    src = []
    tgt = []
    with open(src_dir, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if "：" in line:
                l = line.split("：")  #line eg:"初夏：五言绝句"
                #if there are more than one '：', get title before the first '：'
                title, style = l[0], l[-1]
                if len(title) > 20:
                    title = title[:20]  #cut the longer title
                line = "：".join([title, style])
            src.append(line)

    with open(tgt_dir, 'r', encoding='utf-8') as f:
        for line in f:
            tgt.append(line.strip())
    assert len(src) == len(tgt), 'lines not equal!'
    return src, tgt


auto_loader = AutoLoader("seq2seq",
                         model_name="GLM-large-ch",
                         model_dir=model_dir)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
# Custom model and tokenizer:
# model = GLMForSeq2Seq.from_pretrain(download_path=model_dir,model_name='GLM-large-ch')
# tokenizer = GLMLargeChTokenizer()
trainer = Trainer(
    env_type="pytorch",  #pytorch or deepspeed
    experiment_name="glm_seq2seq",
    batch_size=64,  #96
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=2e-8,  #1e-3
    epochs=100,
    log_interval=10,
    tensorboard_dir="tbsummary",
    eval_interval=2000000,  #no evaluation metric
    load_dir="",
    save_dir="checkpoints_poetry",
    save_interval=1,
    num_checkpoints=1,
    pytorch_device='cuda',
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    hostfile='./hostfile',
    deepspeed_config='./deepspeed.json',
    training_script=__file__,
)


class BertSeq2seqDataset(Dataset):

    def __init__(self, sents_src, sents_tgt):
        super(BertSeq2seqDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

    def __getitem__(self, i):
        source_text = self.sents_src[i]
        target_text = self.sents_tgt[i]
        data = tokenizer.encode_plus(source_text, target_text=target_text)
        return data

    def __len__(self):
        return len(self.sents_src)


class GLMPoetryDynamicCollateFN():  #padding process in each batch

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_token(self, tokens, max_length):
        pad_len = max_length - len(tokens)
        tokens += [self.pad_id] * pad_len
        return tokens

    def pad_position_ids(self, position_ids, max_length):
        pad_len = max_length - len(position_ids[0])
        position_ids[0] += [len(position_ids[0]) + x for x in range(pad_len)]
        position_ids[1] += [1] * pad_len
        return position_ids

    def pad_loss_mask(self, loss_mask, max_length):
        pad_len = max_length - len(loss_mask)
        loss_mask += [0] * pad_len
        return loss_mask

    def __call__(self, batch):
        input_ids = [data["input_ids"] for data in batch]
        target_ids = [data["target_ids"] for data in batch]
        position_ids = [data["position_ids"] for data in batch]
        attention_mask = [data['attention_mask'] for data in batch]
        loss_mask = [data['loss_mask'] for data in batch]

        max_length = max([len(t) for t in input_ids])
        for i in range(len(input_ids)):
            input_ids[i] = self.pad_token(input_ids[i], max_length)
            target_ids[i] = self.pad_token(target_ids[i], max_length)
            position_ids[i] = self.pad_position_ids(position_ids[i],
                                                    max_length)
            loss_mask[i] = self.pad_loss_mask(loss_mask[i], max_length)
        return {
            'input_ids': torch.LongTensor(input_ids),
            'target_ids': torch.LongTensor(target_ids),
            'position_ids': torch.LongTensor(position_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'loss_mask': torch.LongTensor(loss_mask)
        }


train_src, train_tgt = read_file()
print('-----------train data length:', len(train_src))
my_collate_fn = GLMPoetryDynamicCollateFN(
    pad_id=tokenizer.get_command_id('pad'))
train_dataset = BertSeq2seqDataset(train_src, train_tgt)

trainer.train(model, train_dataset=train_dataset, collate_fn=my_collate_fn)
