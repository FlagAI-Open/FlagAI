import torch
from safetensors.torch import load_file, save_file
import os, glob, shutil
import re
from typing import List
import argparse
import numpy as np
from transformers import AutoConfig

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, required=True, help="Path to the original model")
parser.add_argument("--quant-path", type=str, required=True, help="Path to the AutoGPTQ model")
parser.add_argument("--output-path", type=str, required=True, help="Path to save the converted model")

# Copied from https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/auto_gptq/nn_modules/qlinear/qlinear_marlin.py
def get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

PERM, SCALE_PERM, SCALE_PERM_SINGLE = get_perms()

def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:

    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(SCALE_PERM)))[:, SCALE_PERM]
    else:
        s = s.reshape((-1, len(SCALE_PERM_SINGLE)))[:, SCALE_PERM_SINGLE]
    s = s.reshape((-1, size_n)).contiguous()

    return s

def marlin_repack_qweight(qweight: torch.Tensor, bits: int, size_k: int, size_n: int, tile: int = 16) -> torch.Tensor:

    # unpack
    compress_factor = 32 // bits
    mask = (1 << bits) - 1
    qweight = qweight.cpu().numpy().astype(np.uint32)
    unpacked_qweight = np.zeros((size_k, size_n), dtype=np.uint32)
    unpacked_offset = np.arange(size_k) // compress_factor
    unpacked_shift = (np.arange(size_k) % compress_factor) * bits
    unpacked_qweight = (qweight[unpacked_offset, :] >> unpacked_shift[:, None]) & mask

    # permute
    unpacked_qweight = torch.from_numpy(unpacked_qweight.astype(np.int32))
    unpacked_qweight = unpacked_qweight.reshape((size_k // tile, tile, size_n // tile, tile))
    unpacked_qweight = unpacked_qweight.permute(0, 2, 1, 3)
    unpacked_qweight = unpacked_qweight.reshape((size_k // tile, size_n * tile)).contiguous()
    unpacked_qweight = unpacked_qweight.reshape((-1, PERM.numel()))[:, PERM].reshape(unpacked_qweight.shape)

    # repack
    repacked_qweight = np.zeros((unpacked_qweight.shape[0], unpacked_qweight.shape[1] // compress_factor), dtype=np.uint32)
    unpacked_qweight = unpacked_qweight.cpu().numpy().astype(np.uint32)
    for i in range(compress_factor):
        repacked_qweight |= unpacked_qweight[:, i::compress_factor] << (bits * i)
    repacked_qweight = torch.from_numpy(repacked_qweight.astype(np.int32))

    return repacked_qweight

def convert_w4a16_checkpoint(orig_model_path, quant_path, output_path):

    config = AutoConfig.from_pretrained(quant_path)
    
    group_size = config.quantization_config['group_size']
    assert group_size in [-1, 128], "Only group_size -1 and 128 are supported for marlin"

    bits = config.quantization_config['bits']
    assert bits == 4, "Only 4-bit quantization is supported for marlin"
    
    model_path = glob.glob(os.path.join(quant_path, "*.safetensors"))[0]

    autogptq_weigths = load_file(model_path)

    gptq_convert_dict = {
        "model.layers.{}.self_attn.q_proj.qweight": ["model.layers.{}.self_attn.q_proj.scales", "model.layers.{}.self_attn.q_proj.g_idx", "model.layers.{}.self_attn.q_proj.qzeros"], 
        "model.layers.{}.self_attn.k_proj.qweight":["model.layers.{}.self_attn.k_proj.scales", "model.layers.{}.self_attn.k_proj.g_idx", "model.layers.{}.self_attn.k_proj.qzeros"],
        "model.layers.{}.self_attn.v_proj.qweight":["model.layers.{}.self_attn.v_proj.scales", "model.layers.{}.self_attn.v_proj.g_idx", "model.layers.{}.self_attn.v_proj.qzeros"],
        "model.layers.{}.self_attn.o_proj.qweight":["model.layers.{}.self_attn.o_proj.scales", "model.layers.{}.self_attn.o_proj.g_idx", "model.layers.{}.self_attn.o_proj.qzeros"],
        "model.layers.{}.mlp.gate_proj.qweight":["model.layers.{}.mlp.gate_proj.scales", "model.layers.{}.mlp.gate_proj.g_idx", "model.layers.{}.mlp.gate_proj.qzeros"],
        "model.layers.{}.mlp.up_proj.qweight": ["model.layers.{}.mlp.up_proj.scales", "model.layers.{}.mlp.up_proj.g_idx", "model.layers.{}.mlp.up_proj.qzeros"],
        "model.layers.{}.mlp.down_proj.qweight": ["model.layers.{}.mlp.down_proj.scales", "model.layers.{}.mlp.down_proj.g_idx", "model.layers.{}.mlp.down_proj.qzeros"],
        "fc.qweight": ["fc.scales", "fc.g_idx", "fc.qzeros"],
    }

    convert_checkpoint = {}
    processed_keys = set()

    for gptq_key in autogptq_weigths:
        if gptq_key in processed_keys:
            continue
        elif "layers" in gptq_key:
            abstract_key = re.sub(r'(\d+)', '{}', gptq_key)
            layer_num = re.search(r'\d+', gptq_key).group(0)
            if "q_proj" in abstract_key:
                if abstract_key.endswith('qweight'):
                    k_key = gptq_key.replace('q_proj', 'k_proj')
                    v_key = gptq_key.replace('q_proj', 'v_proj')
                    
                    q_weight = autogptq_weigths[gptq_key].clone().cuda()
                    k_weight = autogptq_weigths[k_key].clone().cuda()
                    v_weight = autogptq_weigths[v_key].clone().cuda()
                    x = torch.cat([q_weight, k_weight, v_weight], dim=-1)
                    
                    shape_0 = x.shape[0] * 8
                    shape_1 = x.shape[1]
                    x = marlin_repack_qweight(x, bits, shape_0, shape_1)
                    
                    convert_checkpoint[gptq_key.replace("q_proj", "qkv_proj")] = x.cpu()

                    processed_keys.add(gptq_key)
                    processed_keys.add(k_key)
                    processed_keys.add(v_key)

                    for q_keys in gptq_convert_dict[abstract_key]:
                        if q_keys.endswith("scales"):
                            k_q_keys = q_keys.replace("q_proj", "k_proj")
                            v_q_keys = q_keys.replace("q_proj", "v_proj")   

                            scales_x_q = autogptq_weigths[q_keys.format(layer_num)].clone().cuda()
                            scales_x_k = autogptq_weigths[k_q_keys.format(layer_num)].clone().cuda()
                            scales_x_v = autogptq_weigths[v_q_keys.format(layer_num)].clone().cuda()
                            scales_x = torch.cat([scales_x_q, scales_x_k, scales_x_v], dim=-1)
                            scales_x.data = marlin_permute_scales(scales_x.data.contiguous(),
                                    size_k=shape_0,
                                    size_n=shape_1,
                                    group_size=group_size)
                            convert_checkpoint[q_keys.format(layer_num).replace("q_proj", "qkv_proj")] = scales_x.cpu()
                        
                        processed_keys.add(q_keys.format(layer_num))
                        processed_keys.add(q_keys.replace("q_proj", "k_proj").format(layer_num))
                        processed_keys.add(q_keys.replace("q_proj", "v_proj").format(layer_num))
            elif "gate_proj" in abstract_key:
                if abstract_key.endswith('qweight'):
                    up_key = gptq_key.replace('gate_proj', 'up_proj')
                    
                    gate_weight = autogptq_weigths[gptq_key].clone().cuda()
                    up_weight = autogptq_weigths[up_key].clone().cuda()

                    x = torch.cat([gate_weight, up_weight], dim=-1)
                    
                    shape_0 = x.shape[0] * 8
                    shape_1 = x.shape[1]
                    x = marlin_repack_qweight(x, bits, shape_0, shape_1)
                
                    convert_checkpoint[gptq_key.replace("gate_proj", "gate_up_proj")] = x.cpu()

                    processed_keys.add(gptq_key)
                    processed_keys.add(up_key)

                    for q_keys in gptq_convert_dict[abstract_key]:
                        if q_keys.endswith("scales"):
                            up_q_keys = q_keys.replace("gate_proj", "up_proj")

                            scales_x_gate = autogptq_weigths[q_keys.format(layer_num)].clone().cuda()
                            scales_x_up = autogptq_weigths[up_q_keys.format(layer_num)].clone().cuda()
                            scales_x = torch.cat([scales_x_gate, scales_x_up], dim=-1)
                            scales_x.data = marlin_permute_scales(scales_x.data.contiguous(),
                                    size_k=shape_0,
                                    size_n=shape_1,
                                    group_size=group_size)
                            convert_checkpoint[q_keys.format(layer_num).replace("gate_proj", "gate_up_proj")] = scales_x.cpu()
                        
                        processed_keys.add(q_keys.format(layer_num))
                        processed_keys.add(q_keys.replace("gate_proj", "up_proj").format(layer_num))

            elif "down_proj" in abstract_key or "o_proj" in abstract_key:
                if abstract_key.endswith('qweight'):
                    x = autogptq_weigths[gptq_key].clone().cuda()

                    shape_0 = x.shape[0] * 8
                    shape_1 = x.shape[1]
                    x = marlin_repack_qweight(x, bits, shape_0, shape_1)
                
                    convert_checkpoint[gptq_key] = x.cpu()

                    processed_keys.add(gptq_key)

                    for q_keys in gptq_convert_dict[abstract_key]:
                        if q_keys.endswith("scales"):

                            scales_x = autogptq_weigths[q_keys.format(layer_num)].clone().cuda()
                            scales_x.data = marlin_permute_scales(scales_x.data.contiguous(),
                                    size_k=shape_0,
                                    size_n=shape_1,
                                    group_size=group_size)
                            convert_checkpoint[q_keys.format(layer_num)] = scales_x.cpu()

                        processed_keys.add(q_keys.format(layer_num))

            elif "post_attention_layernorm" in gptq_key or "input_layernorm" in gptq_key:
                convert_checkpoint[gptq_key] = autogptq_weigths[gptq_key].clone()
        elif "fc" in gptq_key and autogptq_weigths[gptq_key].dtype == torch.int32:
            if gptq_key.endswith('qweight'):
                fc_qweight = autogptq_weigths[gptq_key].clone().cuda()
                packed_in_features_x_2, out_features = fc_qweight.shape
                packed_in_features = packed_in_features_x_2 // 2
                in_features = packed_in_features * 32 // bits
                fc1_weight = fc_qweight[:packed_in_features, :].contiguous()
                fc2_weight = fc_qweight[packed_in_features:, :].contiguous()

                fc1_weight = marlin_repack_qweight(fc1_weight, bits, in_features, out_features)
                fc2_weight = marlin_repack_qweight(fc2_weight, bits, in_features, out_features)

                convert_checkpoint[gptq_key] = torch.cat([fc1_weight, fc2_weight], dim=-1).cpu()
                processed_keys.add(gptq_key)

                for fc_key in gptq_convert_dict[gptq_key]:
                    if fc_key.endswith("scales"):
                        fc_scales = autogptq_weigths[gptq_key.replace("qweight", "scales")].clone().cuda()
                        fc_scales_1 = fc_scales[:in_features // group_size, :].contiguous()
                        fc_scales_2 = fc_scales[in_features // group_size:, :].contiguous()

                        fc_scales_1 = marlin_permute_scales(
                            fc_scales_1.data.contiguous(), 
                            size_k=in_features,
                            size_n=out_features,
                            group_size=group_size
                        )
                        fc_scales_2 = marlin_permute_scales(
                            fc_scales_2.data.contiguous(), 
                            size_k=in_features,
                            size_n=out_features,
                            group_size=group_size
                        )
                        # convert_checkpoint[q_keys.format(layer_num).replace("gate_proj", "gate_up_proj")] = scales_x.cpu()
                        convert_checkpoint[gptq_key.replace("qweight", "scales")] = torch.cat([fc_scales_1, fc_scales_2], dim=-1).cpu()
                        processed_keys.add(gptq_key.replace("qweight", "scales"))
        else:  
            convert_checkpoint[gptq_key] = autogptq_weigths[gptq_key].clone()

    save_file(convert_checkpoint, os.path.join(output_path, f"model_gptq.safetensors"))
    # copy quantization config
    config_list = glob.glob(os.path.join(quant_path, "*config.json"))
    for config_file in config_list:
        # copy config to output path
        config_filename = os.path.basename(config_file)
        dst_path = os.path.join(output_path, config_filename)
        shutil.copy2(config_file, dst_path)
    
    # copy tokenizer
    tokenizer_list = glob.glob(os.path.join(orig_model_path, "tokenizer*"))
    for tokenizer_file in tokenizer_list:
        # copy config to output path
        tokenizer_filename = os.path.basename(tokenizer_file)
        dst_path = os.path.join(output_path, tokenizer_filename)
        shutil.copy2(tokenizer_file, dst_path)
    
    # copy "special_tokens_map.json"
    special_tokens_map_file = glob.glob(os.path.join(orig_model_path, "special_tokens_map.json"))[0]
    special_tokens_map_basename = os.path.basename(special_tokens_map_file)
    dst_path = os.path.join(output_path, special_tokens_map_basename)
    shutil.copy2(special_tokens_map_file, dst_path)
    
if __name__=="__main__":
    
    args = parser.parse_args()
    orig_model_path = args.model_path
    quant_path = args.quant_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    convert_w4a16_checkpoint(orig_model_path, quant_path, output_path)