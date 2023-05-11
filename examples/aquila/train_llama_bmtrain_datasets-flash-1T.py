# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
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

#torch.autograd.set_detect_anomaly(True)

from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_datasets
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_weighted_datasets
from gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="llama",
    model_name="llama-7b-en",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=100,
    log_interval=1,
    eval_interval=5000,
    num_gpus=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_llama",
    checkpoint_activations=False,
    save_interval=5000,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()
#env_args.wandb = False
# overwrite
if env_args.yaml_config:
    import yaml
    file_data = open(env_args.yaml_config, 'r', encoding="utf-8").read()
    data = yaml.load_all(file_data)
    delattr(env_args, 'yaml_config')
    arg_dict = env_args.__dict__
    for subdata in data:
        for key, value in subdata.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
trainer = EnvTrainer(env_args)
print(f"*****************{trainer.rank}******************")
# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

print(f"Trainer effective env_args={env_args} local_rank={trainer.local_rank}", flush=True)

## TODO
checkpoints = "/data/ldwang/state_dict/"
model_name = env_args.model_name
print('*'*20, "model_name", model_name, flush=True)

'''
auto_loader = AutoLoader(
    "lm",
    model_name=model_name,
    model_dir=checkpoints,
    only_download_config=True,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
print('*'*20, "model", model)
trainer.pre_train(model)
print('*'*20, "model", model)

'''
cache_dir = checkpoints + model_name
#print('*'*20, "cache_dir", cache_dir)
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#print('*'*20, "tokenizer", tokenizer)

config_file = cache_dir + "/config.json"
# avoid sync loading models in case of Mem OOM
if env_args.bmt_async_load:
    import time
    time.sleep(4.5*60*(trainer.local_rank%8))

# from flagai.model.llama_model import LLAMAModel
# model = LLAMAModel.init_from_json(config_file=config_file)
#model_name='7B'
#if '7b' in env_args.experiment_name.lower():
#    model_name = '7B'
#checkpoint_path = '/data/ldwang/state_dict/Aquila-7b/'
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

#config.bmt = False 
#config.prenorm_std = False 
#config.prenorm = False 
config.use_flash_attn = True
print(config)
model = GPTLMHeadModel(config, device='cpu')#,dtype= torch.float16)
print('*'*20, "model", model)
checkpoint_path = os.path.join('/data2/checkpoints/Aquila-7b-24n8g-V3/16000', "pytorch_model.bin")
ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#ckpt = remap_state_dict_meta_llama(ckpt, config)
model.load_state_dict(ckpt, strict=True)
#del ckpt
#model.load_weights(checkpoint_path)
gc.collect()
torch.cuda.empty_cache()
trainer.pre_train(model)
# model.transformer.bmt_replace()
print('*'*20, "model", model, flush=True)
gc.collect()
torch.cuda.empty_cache()
if True:
    batch1_tok100k = '/data/indexed_dataset/batch1_batch2_tok100k/batch1_tok100k'
    batch2_tok100k = '/data/indexed_dataset/batch1_batch2_tok100k/batch2_tok100k'
    data_prefix = [
        2.242990654,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-openwebtext2_text_document'),
        5.046728972,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-pile-cc_text_document'),
        13.64485981,
        os.path.join(batch1_tok100k, 'wudao-9_text_document'),
        2.336448598,
        os.path.join(batch1_tok100k, 'code_dedup-md5-pile-github_text_document'),
        1.869158879,
        os.path.join(batch1_tok100k, 'codegeex_text_document'),
        1.588785047,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-wikipedia_en_text_document'),
        2.336448598,
        os.path.join(batch1_tok100k, 'cn_baike_text_document'),
        4.205607477,
        os.path.join(batch1_tok100k, 'pile-books_text_document'),
        0.186915888,
        os.path.join(batch1_tok100k, 'cn_ebook_merge_maxlen_text_document'),
        2.429906542,
        os.path.join(batch1_tok100k, 'pile-papers_text_document'),
        1.869158879,
        os.path.join(batch1_tok100k, 'en_dedup-md5-pile-stackexchange_text_document'),
        0.747663551,
        os.path.join(batch1_tok100k, 'cn_zhihu_text_document'),

        31.77570093,
        os.path.join(batch2_tok100k, 'ccnews_text_document'),
        12.42990654,
        os.path.join(batch2_tok100k, 'c4_text_document'),
        11.58878505,
        os.path.join(batch2_tok100k, 'wudao-3-8_text_document'),
        1.869158879,
        os.path.join(batch2_tok100k, 'hf-wiki_text_document'),
        0.654205607,
        os.path.join(batch2_tok100k, 'sjt_text_document'),
        1.214953271,
        os.path.join(batch2_tok100k, 'col_text_document'),
        1.121495327,
        os.path.join(batch2_tok100k, 'byg-cn_text_document'),
        0.093457944,
        os.path.join(batch2_tok100k, 'qa_text_document'),
        0.747663551,
        os.path.join(batch2_tok100k, 'wenge-zhihu-high_text_document'),
    ]

    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    splits_string = '9999,1'
    ## 2. specify total samples needed
    ## 400B = 400 * 1000 * 1000 * 1000./ 2048 = 195312500
    ## 1000B = 1000 * 1000 * 1000 * 1000./ 2048 = 488281250
    train_max_num_samples = 195312500
    train_max_num_samples = 488281250
    ## 1070B
    train_max_num_samples = 522460937
    train_valid_test_num_samples = [train_max_num_samples, int(train_max_num_samples*0.00001)]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    train_dataset, valid_dataset, _ = _build_train_valid_test_weighted_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup,
        train_max_num_samples)
    print("Total train_dataset: ", len(train_dataset), flush=True)
    print("Total valid_dataset: ", len(valid_dataset), flush=True)
    def forward_step(data, model, mems=None):
        """Simple forward step. """
        # data['mems'] = mems
        model_output = model(data['input_ids'])
        # print(model_output)
        logits = model_output.logits
        loss = model_output.loss
        hidden_states = None
        if 'hidden_states' in model_output:
            hidden_states = model_output['hidden_states']
        elif 'encoder_hidden_states' in model_output:
            hidden_states = model_output['encoder_hidden_states']

        return {
            'loss': loss,
            'hidden_states': hidden_states,
            'logits': logits.contiguous().float()
        }
    def collate_fn(batch):
        def padding(indice, max_length, pad_idx=tokenizer.token_end_id):
            pad_indice = [
                item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
            ]
            return torch.tensor(pad_indice)
    
        input_ids = [data["input_ids"] for data in batch]
        max_length = max([len(t) for t in input_ids])
        input_ids = padding(input_ids, max_length)[:,:seq_length]
    
        data = {
            "input_ids": input_ids
            # "labels": input_ids
        }
        return data
    trainer.forward_step = forward_step
    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=collate_fn,
        optimizer=None,
        rank_split=False)

