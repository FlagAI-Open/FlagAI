from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os
import types

import torch

import argparse


def get_function_name(question: str, lang: str = 'Python'):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def entry_point(
    problem_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )

    return results


def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def gen_prompt(prompt: str, args) -> str:
    prompt = (
        "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
        + prompt
    )
    return prompt


def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)


def test_fix_indents():
    text = "   # TODO: Implement separate_paren_groups\nreturn []"
    print(fix_indents(text))


def evaluate(model, data_path: str, args, **kwargs) -> dict:
    dataset = read_problems(data_path)
    n_sample = kwargs.get("n_sample", 1)
    samples = []
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")

    for task_id in dataset:
        for i in range(n_sample):
            prompt = dataset[task_id]["prompt"]
            prompt = gen_prompt(prompt, args)
            completion = model.run(prompt)
            completion = fix_indents(completion)
            sample = dict(task_id=task_id, completion=filter_code(completion))

            samples.append(sample)
            progress_bar.update(1)

    progress_bar.close()

    result = None

    pred_filename = f"humaneval_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(problem_file=data_path, sample_file=pred_filename)
    return result

class AutoRegressiveModel:
    def __init__(self, target_model, tokenizer, max_len):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def run(self, prompt):
        enc = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        input_ids = enc['input_ids']
        prompt_len = input_ids.shape[-1]
        output = self.target_model.generate(**enc, max_new_tokens=self.max_len, do_sample=True, temperature=0.1, top_p=0.95)
        output = output[:,prompt_len:]
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../datas/HumanEval.jsonl")
    parser.add_argument('--generate_len', type=int, help='Generate length during testing', default=512) 
    parser.add_argument("--lora_type", type=str, help="calora or qlora or lora")
    args = parser.parse_args()  
    return args


def main():
    args = parse()

    model_name = "../models/llama-2-13b-hf"

    modified_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

    # Load the model and tokenizer
    if args.lora_type == 'calora':
        print(f"results/llama-2-13b-calora/{os.environ['CHECKPOINT_VERSION']}")
        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "../models/llama-2-13b-hf",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # use_cache=False,
            # use_flash_attention_2=True,
            device_map="auto",
        )
        peft_config = LoraConfig(
            target_modules=modified_modules,
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config, adapter_name="inherit")
        model.add_adapter("recover", peft_config)
        model.load_adapter(f"results/llama-2-13b-calora/{os.environ['CHECKPOINT_VERSION']}/inherit", "inherit")
        model.load_adapter(f"results/llama-2-13b-calora/{os.environ['CHECKPOINT_VERSION']}/recover", "recover")
        def recover_func(self, x: torch.Tensor):
            import torch.nn.functional as F
            result = self._forward(x)

            result += (
                self.lora_B["recover"](
                    self.lora_A["recover"](self.lora_dropout["recover"](
                        x.to(self.lora_A["recover"].weight.dtype)
                    ))
                )
                * self.scaling["recover"]
            )
            return result
        for n, m in model.named_modules():
            if n.split('.')[-1] in modified_modules:
                m._forward = m.forward
                m.forward = types.MethodType(recover_func, m) # hack in to lora
    elif args.lora_type == 'calora-inherit-only':
        print(f"results/llama-2-13b-lora/{os.environ['CHECKPOINT_VERSION']}")
        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "../models/llama-2-13b-hf",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # use_cache=False,
            # use_flash_attention_2=True,
            device_map="auto",
        )
        peft_config = LoraConfig(
            target_modules=modified_modules,
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config, adapter_name="inherit")
        model.load_adapter(f"results/llama-2-13b-lora/{os.environ['CHECKPOINT_VERSION']}", "inherit")
    elif args.lora_type == 'qlora':
        print(f"results/llama-2-13b-qlora/{os.environ['CHECKPOINT_VERSION']}")
        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "../models/llama-2-13b-hf",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # use_cache=False,
            # use_flash_attention_2=True,
            device_map="auto",
        )
        peft_config = LoraConfig(
            target_modules=modified_modules,
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.load_adapter(f"results/llama-2-13b-qlora/{os.environ['CHECKPOINT_VERSION']}", "default")
    elif args.lora_type == 'lora':
        print(f"results/llama-2-13b-lora/{os.environ['CHECKPOINT_VERSION']}")
        model = AutoModelForCausalLM.from_pretrained(
            "../models/llama-2-13b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        peft_config = LoraConfig(
            target_modules=modified_modules,
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.load_adapter(f"results/llama-2-13b-lora/{os.environ['CHECKPOINT_VERSION']}", "default")
    elif args.lora_type == 'none':
        model = AutoModelForCausalLM.from_pretrained(
            "../models/llama-2-13b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif args.lora_type == 'q':
        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "../models/llama-2-13b-hf",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # use_cache=False,
            # use_flash_attention_2=True,
            device_map="auto",
        )
    else:
        raise NotImplementedError()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoRegressiveModel(target_model=model, tokenizer=tokenizer, max_len=args.generate_len)

    # test human_eval
    result = evaluate(model, args.data_path, args)
    print(result)


if __name__ == "__main__":
    main()
