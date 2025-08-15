from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

torch.manual_seed(0)

def llm_load(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

    return model, tokenizer

def convert_llm():
    # model.embed_tokens.weight * scale_emb
    state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"] * scale_emb

    # lm_head.weight / (hidden_size / dim_model_base)
    state_dict["lm_head.weight"] = state_dict["lm_head.weight"] / (hidden_size / dim_model_base)

    for i in range(num_layers):
        attn_out_name = f"model.layers.{i}.self_attn.o_proj.weight"
        state_dict[attn_out_name] = state_dict[attn_out_name] * (scale_depth / math.sqrt(num_layers))

        ffn_down_proj_name = f"model.layers.{i}.mlp.down_proj.weight"
        state_dict[ffn_down_proj_name] = state_dict[ffn_down_proj_name] * (scale_depth / math.sqrt(num_layers))

    torch.save(state_dict, "./pytorch_model.bin")

if __name__ == "__main__":
    model, tokenizer = llm_load("/DATA/disk0/zhaoweilun/minicpm4/models/stable_7T_decay_700B_decay2_300B_longdecay_1sw1fa_sft_50B_release")

    scale_emb = model.config.scale_emb
    dim_model_base = model.config.dim_model_base
    scale_depth = model.config.scale_depth
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"scale_emb = {scale_emb}")
    print(f"dim_model_base = {dim_model_base}")
    print(f"scale_depth = {scale_depth}")
    print(f"num_layers = {num_layers}")
    print(f"hidden_size = {hidden_size}")

    state_dict = model.state_dict()
    convert_llm()