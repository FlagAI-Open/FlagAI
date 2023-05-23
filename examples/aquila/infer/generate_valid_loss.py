import os
import sys
sys.path.append("/data2/gitee_test/flagai-internal")
import torch
from torch.utils.data import Dataset
import gc
gc.collect()
torch.cuda.empty_cache()
import flash_attn
# from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.env_args import EnvArgs
from flagai.env_trainer_v1 import EnvTrainer
from tqdm import tqdm
#torch.autograd.set_detect_anomaly(True)

from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_datasets
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_weighted_datasets
from gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint
checkpoints = '/data2/state_dict/'
model_name = 'Aquila-30b'
cache_dir = checkpoints + model_name
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoints, model_name))
config.vocab_size=100008
config.use_cache = False
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.layer_norm_epsilon = 1e-5

config.fused_bias_fc = False
config.fused_mlp = False  # We don't have fused GatedMLP yet
config.fused_dropout_add_ln = False
config.residual_in_fp32 = False
config.use_flash_attn = True
print(config)

import jsonlines
import numpy as np
conversations = []
max_seq_len=2048
with jsonlines.open("/data/benchmark/common/wikitext/wikitext.jsonl") as reader:
    for line in reader:
        if len(line['text']) < 100:
            continue
        obj = dict()
        obj['text'] = line['text']
        conversations.append(obj)

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, maxlen=512):
        super(ConversationDataset, self).__init__()
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        text = self.conversations[i]['text']

            # chat_desc
        example = self.tokenizer.encode_plus(f"{text}", None, max_length=None)['input_ids']
        EOS_TOKEN = example[-1]
        example = example[:-1] # remove eos
        import copy
        labels = copy.deepcopy(example)
        example.append(EOS_TOKEN)
        labels.append(EOS_TOKEN)
        
        output = {
            "input_ids": torch.LongTensor(example).unsqueeze(0).cuda(),
            "labels": torch.LongTensor(labels).unsqueeze(0).cuda(),
        }
        return output

    def __len__(self):
        return len(self.conversations)

    @staticmethod
    def collate_fn(batch):
        def padding(indice, max_length, pad_idx=0):
            pad_indice = [
                item + [pad_idx] * max(0, max_length - len(item)) for item in indice
            ]
            return torch.tensor(pad_indice)

        input_ids = [data["input_ids"] for data in batch]
        labels = [data["labels"] for data in batch]
        max_length = max_seq_len
        input_ids = padding(input_ids, max_length)[:,:max_length]
        labels = padding(labels, max_length, pad_idx=env_args.IGNORE_INDEX)[:,:max_length]

        data = {
            "input_ids": input_ids,
            "labels": labels
        }
        return data

data_len = len(conversations)
train_size = data_len
train_conversations = conversations[:train_size]
train_dataset = ConversationDataset(train_conversations,
                                    tokenizer=tokenizer,
                                    maxlen=max_seq_len)

model_list =[1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
model = GPTLMHeadModel(config, device='cuda:0', dtype=torch.float16)
for ckpt in model_list:
    #checkpoint_path = os.path.join(f'/data2/state_dict/Aquila-7b-67000', "pytorch_model.bin")
    #ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_path = os.path.join(f'/data/benchmark/Aquila30B_17May/{ckpt}', "pytorch_model.bin")
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))['module']
    model.load_state_dict(ckpt, strict=True)
    print(f"eval on ckpt {ckpt}...")
    gc.collect()
    torch.cuda.empty_cache()
    losses = []
    for d in tqdm(train_dataset):
        try:
            output = model.forward(**d)
        except Exception as e:
            continue
        losses += output.loss.view(-1).detach().cpu().numpy().tolist()
    print(sum(losses)/len(losses))
