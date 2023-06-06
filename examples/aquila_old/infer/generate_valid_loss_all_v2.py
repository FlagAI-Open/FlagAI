import os
import sys
sys.path.append("/data2/gitee_infer/flagai-internal")
import torch
from torch.utils.data import Dataset
import gc
import json
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
torch.cuda.set_device("cuda:0")
with jsonlines.open("/data/benchmark/package/benchmark_v5.jsonl") as reader:
    for line in reader:
        d = {"input_ids":torch.LongTensor(line[:2048]).unsqueeze(0).cuda()}
        conversations.append(d)

model_list = [ckpt]
model = GPTLMHeadModel(config, device='cuda:0', dtype=torch.float16)
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
    for d in tqdm(conversations):
        try:
            output = model.forward(**d)
        except Exception as e:
            continue
        losses += output.loss.view(-1).detach().cpu().numpy().tolist()
        accuracy += output.acc.view(-1).detach().cpu().numpy().tolist()
#print(f"{ckpt} {sum(losses)/len(losses)})
with open("all_loss.log",'a') as wf:
    wf.write(f"{ckpt} {sum(losses)/len(losses)} {sum(accuracy)/len(losses)}\n")
