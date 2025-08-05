import os
import torch
import argparse
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument("--fp-model-path", type=str, required=True, help="Path to the fp model")
parser.add_argument("--quant-model-path", type=str, required=True, help="Path to the quantized model")
parser.add_argument("--output-path", type=str, required=True, help="Path to save the converted model")

def post_process_eagle_w4a16_ckpt(fp_model_path, quant_model_path, output_path):
    fp_model = torch.load(os.path.join(fp_model_path, "pytorch_model.bin"))
    quant_model = load_file(os.path.join(quant_model_path, "model_gptq.safetensors"))

    new_state_dict = {}

    assert (fp_model["embed_tokens.weight"].to(torch.float16) == quant_model["model.embed_tokens.weight"].cuda().to(torch.float16)).all(), "embed_tokens.weight mismatch"
    new_state_dict["embed_tokens.weight"] = fp_model["embed_tokens.weight"].to(torch.float16)
    
    if "fc.weight" in quant_model.keys():
        assert (fp_model["fc.weight"].to(torch.float16) == quant_model["fc.weight"].cuda().to(torch.float16)).all(), "fc.weight mismatch"
        new_state_dict["fc.weight"] = fp_model["fc.weight"].to(torch.float16)
    elif "fc.qweight" in quant_model.keys():
        new_state_dict["fc.qweight"] = quant_model["fc.qweight"]
        new_state_dict["fc.scales"] = quant_model["fc.scales"]

    new_state_dict["input_norm1.weight"] = fp_model["input_norm1.weight"].to(torch.float16)
    new_state_dict["input_norm2.weight"] = fp_model["input_norm2.weight"].to(torch.float16)

    for key, value in quant_model.items():
        if "model.layers." in key:
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value
    
    save_file(new_state_dict, os.path.join(output_path, f"model_gptq.safetensors"))

    os.system(f"cp {quant_model_path}/*.json {output_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    fp_model_path = args.fp_model_path
    quant_model_path = args.quant_model_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    post_process_eagle_w4a16_ckpt(fp_model_path, quant_model_path, output_path)
    