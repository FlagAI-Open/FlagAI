# code based on https://www.philschmid.de/instruction-tune-llama-2

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
from random import randrange
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import TrainingArguments
from trl import SFTTrainer
import argparse
import types
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--lora_type", type=str, help="calora or qlora or lora")
args = parser.parse_args()
 
# Load dataset from the hub
dataset = load_from_disk("../datas/CodeAlpaca-20k/train") # sahil2801/CodeAlpaca-20k
# Or dataset = load_from_disk("../datas/CodeAlpaca-20k")["train"] for some datasets version
 
print(f"dataset size: {len(dataset)}")

def format_instruction(sample):
    if sample['input'] != '':
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Response:
{sample['output']}
"""
 
model_id = "../models/llama-2-13b-hf"
 
# modified_modules = ['q_proj','v_proj']
modified_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

if args.lora_type == 'qlora':
    # BitsAndBytesConfig int-4 config
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
        model_id,
        # load_in_8bit=True,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        # use_cache=False,
        # use_flash_attention_2=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        target_modules=modified_modules,
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    trainer_cls = SFTTrainer
elif args.lora_type == 'lora':
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_cache=False,
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
    trainer_cls = SFTTrainer
elif args.lora_type == 'calora':
    # BitsAndBytesConfig int-4 config
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
        model_id,
        # load_in_8bit=True,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        # use_cache=False,
        # use_flash_attention_2=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        target_modules=modified_modules,
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config, adapter_name="inherit")
    model.load_adapter("results/llama-2-13b-lora/checkpoint-346", "inherit")
    teacher_model = deepcopy(model) 
    model.add_adapter("recover", peft_config)
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
    class DistillTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            inputs['output_hidden_states'] = True
            loss1, outputs = super().compute_loss(model, inputs, return_outputs=True)
            hidden = outputs['hidden_states'][-1]
            with torch.no_grad():
                teacher_hidden = teacher_model(**inputs)['hidden_states'][-1]
            loss_dis = torch.nn.MSELoss()
            loss = loss1 + 0.055 * loss_dis(hidden, teacher_hidden)
            return (loss, outputs) if return_outputs else loss
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    trainer_cls = DistillTrainer
 
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
 
 
args = TrainingArguments(
    output_dir=f"results/llama-2-13b-{args.lora_type}",
    num_train_epochs=5,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True # disable tqdm since with packing values are in correct
)

max_seq_length = 2048 # max sequence length for model and packing of the dataset
 
trainer = trainer_cls(
    model=model,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled
 
# save model
trainer.save_model()
