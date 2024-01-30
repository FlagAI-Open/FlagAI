# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [CreatedDate]  : Thursday, 1970-01-01 08:00:00
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comments]
# ------------------------------------------------------------------
# 将flash框架转成flagai框架


import os
import shutil
import sys

import torch

project_path = "../../../../FlagAI"
sys.path.append(project_path)

from flagai.data.tokenizer import Tokenizer
from flagai.model.aquila_model import AQUILAModel
from flash_attn.models.gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import (config_from_checkpoint,
                                     llama_config_to_gpt2_config,
                                     remap_state_dict_meta_llama,
                                     state_dicts_from_checkpoint)



device = f"cuda"


def init_flagai(config_source_dir):
    # Flagai 模型初始化
    model_flagai = AQUILAModel.init_from_json(config_file=os.path.join(config_source_dir, "config.json"))
    print("model_flagai init ")
    model_flagai.eval()
    return model_flagai

# 加载Aquila7B-V1的ckpt
# ckpt_iter = int(sys.argv[1])
# ckpt_flagai = torch.load(f'{ckpt_iter}/pytorch_model.bin')


def init_flash(model_path):
    # 根据config初始化 Aquila7B 的flash attention 模型

    model_name = '7B'
    config = llama_config_to_gpt2_config(
        config_from_checkpoint(
            "./aquila_flash_config", model_name))
    config.vocab_size = 100008 # 写死了
    config.use_cache = True
    config.attn_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.layer_norm_epsilon = 1e-5 # 这里写死了，和config配置中的值不相关

    config.fused_bias_fc = False
    config.fused_mlp = False  # We don't have fused GatedMLP yet
    config.fused_dropout_add_ln = False
    config.residual_in_fp32 = False
    config.bmt = False
    config.prenorm = True
    config.use_flash_attn = True
    print(config)
    model_flash = GPTLMHeadModel(config, device="cuda", dtype=torch.float16)

    checkpoint_path = os.path.join(f'{model_path}', "pytorch_model.bin")
    ckpt = torch.load(checkpoint_path, map_location="cuda")

    model_flash.load_state_dict(ckpt, strict=True)

    sd = model_flash.state_dict()
    torch.save(sd, 'tmp_model.pt')
    model_flash.eval()

    print("model_flash loaded")

    return model_flash

# 转化  flash attention的权重为flagai的格式
# 输入是只有模型{参数名字：参数}的字典
# 如果是flashai的权重，需要注意取 ckpt['module']


def transform_flash_to_flagai(ckpt):
    print("transform_flash_to_flagai")
    tgt_ckpt = {}
    tgt_ckpt["tok_embeddings.weight"] = ckpt.pop("transformer.embeddings.word_embeddings.weight")
    tgt_ckpt["output.weight"] = ckpt.pop("lm_head.weight")
    tgt_ckpt["norm.weight"] = ckpt.pop("transformer.ln_f.weight")

    for l in range(32):
        # attention
        Wqkv = ckpt.pop(f'transformer.layers.{l}.mixer.Wqkv.weight')
        split_size = Wqkv.size()[0]//3
        Wq, Wk, Wv = torch.split(Wqkv, split_size)
        tgt_ckpt[f'layers.{l}.attention.wq.weight'] = Wq
        tgt_ckpt[f'layers.{l}.attention.wk.weight'] = Wk
        tgt_ckpt[f'layers.{l}.attention.wv.weight'] = Wv

        tgt_ckpt[f'layers.{l}.attention.wo.weight'] = ckpt.pop(f'transformer.layers.{l}.mixer.out_proj.weight')
        # feedforward
        W31 = ckpt.pop(f'transformer.layers.{l}.mlp.fc1.weight')
        split_size = W31.size()[0]//2
        W3, W1 = torch.split(W31, split_size)
        tgt_ckpt[f'layers.{l}.feed_forward.w1.weight'] = W1
        tgt_ckpt[f'layers.{l}.feed_forward.w3.weight'] = W3
        tgt_ckpt[f'layers.{l}.feed_forward.w2.weight'] = ckpt.pop(f'transformer.layers.{l}.mlp.fc2.weight')
        # layernorm
        tgt_ckpt[f"layers.{l}.attention_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm1.weight')
        tgt_ckpt[f"layers.{l}.ffn_norm.weight"] = ckpt.pop(f'transformer.layers.{l}.norm2.weight')
    return tgt_ckpt


def write_flash2flagai(config_source_dir, model_flagai, model_path_flash):

    dest_dir = model_path_flash+"-flash2flagai_v2"+"/"+"aquilachat-7b"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print(f"dest directory is [{dest_dir}]")
    torch.save(model_flagai.state_dict(), f"{dest_dir}/pytorch_model.bin")

    copy_src_to_dest(config_source_dir, dest_dir)
    # shutil.remove("./tmp_model.pt")


def copy_src_to_dest(source, dest):
    for folderpath, folders, files in os.walk(source):
        for _file in files:
            if os.path.join(folderpath, _file) != os.path.join(os.curdir, _file):
                try:
                    src_file = os.path.join(folderpath, _file)
                    dest_file = os.path.join(dest, _file)
                    shutil.copy(src_file, dest_file)
                    print(f"source is [{src_file}], dest is [{dest_file}]")
                except Exception as exc:
                    print(exc)
    return

# copy_src_to_dest( "./configs/flagai-7b","./")


if __name__ == "__main__":

    # flage = sys.argv[1]
    flage = "flash2flagai"
    config_source_dir = "./configs/flagai-7b"

    if flage == "flash2flagai":

        model_path_flash = "checkpoints/1182"

        model_flash = init_flash(model_path_flash)

        model_flagai = init_flagai(config_source_dir)

        checkpoint_path = "tmp_model.pt"
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_flagai.half()

        ckpt_flash2flagai = transform_flash_to_flagai(ckpt)

        model_flagai.load_state_dict(ckpt_flash2flagai, strict=True)

        write_flash2flagai(config_source_dir, model_flagai, model_path_flash)

        os.remove("tmp_model.pt")

    else:
        pass

    print("记得将config文件复制过去")
