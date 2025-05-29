"""
Adapted from https://github.com/GanjinZero/RRHF/blob/main/apply_delta.py
"""

import torch
from fire import Fire
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


def main(base_model_path, target_model_path, delta_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the delta from {delta_path}")
    delta = AutoModelForCausalLM.from_pretrained(
        delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)

    DEFAULT_PAD_TOKEN = "[PAD]"
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    except ValueError:
        base_tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)

    num_new_tokens = base_tokenizer.add_special_tokens(
        dict(pad_token=DEFAULT_PAD_TOKEN)
    )

    base.resize_token_embeddings(len(base_tokenizer))
    input_embeddings = base.get_input_embeddings().weight.data
    output_embeddings = base.get_output_embeddings().weight.data
    input_embeddings[-num_new_tokens:] = 0
    output_embeddings[-num_new_tokens:] = 0

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


"""
p apply_delta.py decapoda-research/llama-7b-hf wombat-7b-gpt4 GanjinZero/wombat-7b-gpt4-delta
"""


if __name__ == "__main__":
    Fire(main)
