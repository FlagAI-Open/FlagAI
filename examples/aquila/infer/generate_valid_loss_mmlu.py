import os
import sys
sys.path.append("/data2/gitee_infer/flagai-internal")
import torch
from torch.utils.data import Dataset
import gc
gc.collect()
torch.cuda.empty_cache()
import flash_attn
# from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from tqdm import tqdm
#torch.autograd.set_detect_anomaly(True)

from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_datasets
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_weighted_datasets
from gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint
checkpoints = '/data2/state_dict/'
model_name = 'Aquila-7b-67000'
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
ckpt=sys.argv[1]
import jsonlines
import numpy as np
conversations = []
max_seq_len=2048
torch.cuda.set_device("cuda:7")
with jsonlines.open("/data/benchmark/package/benchmark_mmlu.jsonl") as reader:
    for line in reader:
        obj = dict()
        text = (line['prompt'] + line['answer']).replace("\n\n",".").replace("\n"," ").replace("Passage: ","").replace("Answer","")
        obj['text'] = text
        conversations.append(obj)

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, maxlen=512):
        super(ConversationDataset, self).__init__()
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        text = self.conversations[i]['text'][:2048]

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

model_list = [ckpt]
model = GPTLMHeadModel(config, device='cuda:7', dtype=torch.float16)
for ck in model_list:
    #checkpoint_path = os.path.join(f'/data2/state_dict/Aquila-7b-67000', "pytorch_model.bin")
    #ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_path = os.path.join(f'{ck}', "pytorch_model.bin")
    ckpt_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt_model, strict=True)
    gc.collect()
    torch.cuda.empty_cache()
    losses = []
    accuracy = []
    model.eval()
    for d in tqdm(train_dataset):
        try:
            output = model.forward(**d)
        except Exception as e:
            continue
        losses += output.loss.view(-1).detach().cpu().numpy().tolist()
        accuracy += output.acc.view(-1).detach().cpu().numpy().tolist()
#print(f"{ckpt} {sum(losses)/len(losses)})
with open("mmlu_loss.log",'a') as wf:
    wf.write(f"{ckpt} {sum(losses)/len(losses)} {sum(accuracy)/len(losses)}\n")
