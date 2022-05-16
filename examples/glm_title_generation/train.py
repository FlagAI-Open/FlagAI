import os
import numpy as np
import torch
from torch.utils.data import Dataset
from flagai.model.glm_model import GLMModel, GLMForSeq2Seq
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(
    env_type="pytorch",
    experiment_name="roberta_seq2seq",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=10,
    log_interval=10,
    eval_interval=10000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints",
    save_epoch=1,
    num_checkpoints=1,
    master_ip='127.0.0.1',
    master_port=17750,
    num_nodes=1,
    num_gpus=2,
    hostfile=
    '/data/liuguang/test_Sailing/Sailing/examples/bert_title_generation/hostfile',
    deepspeed_config=
    '/data/liuguang/test_Sailing/Sailing/examples/bert_title_generation/deepspeed.json',
    training_script=__file__,
)
src_dir = './examples/glm_title_generation/data/train.src'
tgt_dir = './examples/glm_title_generation/data/train.tgt'
model_dir = "./state_dict/roberta/"  # 模型位置
os.makedirs(model_dir, exist_ok=True)
model_save_path = "./bert_auto_title_model.bin"
maxlen = 256
model = GLMForSeq2Seq.from_pretrain(model_name='glm_large_ch')
tokenizer = GLMLargeChTokenizer()


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


class BertSeq2seqDataset(Dataset):

    def __init__(self,
                 sents_src,
                 sents_tgt,
                 tokenizer,
                 max_src_length=300,
                 max_tgt_length=200):
        super(BertSeq2seqDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.no_block_position = False

    def __getitem__(self, i):
        source_text = self.sents_src[i]
        target_text = self.sents_tgt[i]
        cls_id = self.tokenizer.get_command('ENC').Id
        mask_token = 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text)
        prompt = [cls_id, mask_id] + self.tokenizer.EncodeAsIds(" Content:")
        if len(source_tokens) > self.max_src_length - len(prompt):
            source_tokens = source_tokens[:self.max_src_length - len(prompt)]
        source_tokens = prompt + source_tokens

        if len(source_tokens) < self.max_src_length:
            source_tokens = source_tokens + [pad_id] * (self.max_src_length -
                                                        len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)
        target_tokens = self.tokenizer.EncodeAsIds(" " + target_text)
        target_tokens = target_tokens + [eop_id]
        if len(target_tokens) > self.max_tgt_length:
            target_tokens = target_tokens[:self.max_tgt_length]
            target_truncated = True
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < self.max_tgt_length:
            loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
            target_tokens += [pad_id
                              ] * (self.max_tgt_length - len(target_tokens))
        tokens = source_tokens + [sop_id] + target_tokens[:-1]
        loss_mask = [0] * len(source_tokens) + loss_mask
        target_ids = [0] * len(source_tokens) + target_tokens
        position_ids += [mask_pos] * len(target_tokens)
        if self.no_block_position:
            block_position_ids += [1] * len(target_tokens)
        else:
            block_position_ids += list(range(1, len(target_tokens) + 1))
        position_ids = [position_ids, block_position_ids]
        sample = {
            'input_ids': np.array(tokens, dtype=np.int64),
            'target_ids': np.array(target_ids, dtype=np.int64),
            'attention_mask': np.array(sep, dtype=np.int64),
            'loss_mask': np.array(loss_mask, dtype=np.int64),
            "position_ids": np.array(position_ids, dtype=np.int64)
        }
        return sample

    def __len__(self):

        return len(self.sents_src)


sents_src, sents_tgt = read_file()
data_len = len(sents_tgt)
train_size = int(data_len * 0.8)
train_src = sents_src[:train_size][:2000]
train_tgt = sents_tgt[:train_size][:2000]

val_src = sents_src[train_size:]
val_tgt = sents_tgt[train_size:]

train_dataset = BertSeq2seqDataset(train_src,
                                   train_tgt,
                                   tokenizer=tokenizer,
                                   max_src_length=300,
                                   max_tgt_length=200)
val_dataset = BertSeq2seqDataset(val_src,
                                 val_tgt,
                                 tokenizer=tokenizer,
                                 max_src_length=300,
                                 max_tgt_length=200)

trainer.train(model, train_dataset=train_dataset, valid_dataset=val_dataset)
