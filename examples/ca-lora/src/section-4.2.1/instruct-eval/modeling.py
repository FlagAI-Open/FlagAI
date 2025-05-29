import json
import signal
import time
from pathlib import Path
from typing import Optional, Tuple

import openai
import rwkv
import tiktoken
import torch
import torch.nn as nn
import transformers
from fire import Fire
from peft import PeftModel
from pydantic import BaseModel
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from torchvision.datasets.utils import download_url
from transformers import AutoTokenizer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
    LlamaConfig,
    BitsAndBytesConfig,
)

import bmtrain as bmt
from loras_awq import LoraModel
from loras_f import LoraModel as LoraModelQlora
from loras import LoraModel as LoraModelbmt
from model_center.model import Llama, LlamaConfig
from model_center.generation.llama import LlamaBeamSearch
from model_center.tokenizer import LlamaTokenizer as LlamaTokenizerBmt
from bmcook.quant import *
from bmcook.pruning import *
from bmcook.utils import config as bmcook_config

import quant

def set_model_otherparam(model, state: bool):
    for n, p in model.named_parameters():
        if 'lora' not in n:
            p.requires_grad = state

class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    max_input_length: int = 512
    max_output_length: int = 512

    def run(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def count_text_length(self, text: str) -> int:
        raise NotImplementedError

    def check_valid_length(self, text: str) -> bool:
        return self.count_text_length(text) <= self.max_input_length

    def load(self):
        raise NotImplementedError


class OpenAIModel(EvalModel):
    model_path: str
    engine: str = ""
    use_azure: bool = False
    tokenizer: Optional[tiktoken.Encoding]
    api_endpoint: str = "https://research.openai.azure.com/"
    api_version: str = "2023-03-15-preview"
    timeout: int = 60
    temperature: float = 0.0

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # chatgpt/gpt-4

        with open(self.model_path) as f:
            info = json.load(f)
            openai.api_key = info["key"]
            self.engine = info["engine"]

        if self.use_azure:
            openai.api_type = "azure"
            openai.api_base = self.api_endpoint
            openai.api_version = self.api_version

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                key = "engine" if self.use_azure else "model"
                kwargs = {key: self.engine}
                response = openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    request_timeout=self.timeout,
                    temperature=0,  # this is the degree of randomness of the model's output
                    **kwargs,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content
            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("OpenAIModel request failed, retrying.")

        return output

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text))

    def get_choice(self, prompt: str, **kwargs) -> str:
        self.load()

        def handler(signum, frame):
            raise Exception("Timeout")

        signal.signal(signal.SIGALRM, handler)

        for i in range(3):  # try 5 times
            signal.alarm(2)  # 5 seconds
            try:
                response = openai.ChatCompletion.create(
                    engine=self.model_path,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                if "content management policy" in str(e):
                    break
                else:
                    time.sleep(3)
        return "Z"


class SeqToSeqModel(EvalModel):
    model_path: str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        start_token = torch.tensor(
            [[self.tokenizer.pad_token_id]], dtype=torch.long
        ).to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                decoder_input_ids=start_token,
                **kwargs,
            ).logits[0, 0]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


class CausalModel(SeqToSeqModel):
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, **args
            )
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class AwqModel(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, **args
            )
            self.model.eval()
            # self.model = self._set_calora(self.model, pet=True, recover=True, calora_path='path/calora-awq/final/finetune-llama-Alpaca-0-6800_calora.pt')
            # self.model = self._set_calora(self.model, pet=True, recover=False, calora_path='path/calora-awq/inherit-old-hf.pt')
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B
    
    def _set_calora(
        self,
        model,
        # control params
        pet: bool = False,
        recover: bool = False,
        # calora path params
        calora_path: str = "",
    ):
        
        # pet & recover =========================================
        if pet: # add lora
            if recover: # add recover
                delta_model = LoraModel(
                    backbone_model=model,
                    modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                    lora_r=16,
                    backend='bmt',
                    lora_type='full',
                )
                delta2_model = LoraModel(
                    backbone_model=model,
                    modified_modules=['gate_proj', 'up_proj', 'down_proj'],
                    lora_r = 16, # TODO
                    lora_dropout=0.05,
                    backend='bmt',
                    lora_type='activate',
                )
                set_model_otherparam(model, False)
                if bmt.rank() == 0:
                    delta2_model.log()
            else: # without recover
                delta_model = LoraModel(
                    backbone_model=model,
                    modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                    lora_r=16,
                    lora_dropout=0.05,
                    backend='bmt',
                    lora_type='normal',
                )
                set_model_otherparam(model, False)
                
                if bmt.rank() == 0:
                    delta_model.log()
        else: # without lora
            if recover: # add recover
                delta_model = LoraModel(
                    backbone_model=model,
                    modified_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                    lora_r = 16, # TODO
                    lora_dropout=0.05,
                    backend='bmt',
                    lora_type='activate',
                )
                set_model_otherparam(model, False)
                
        # load calora params: there should ONLY be calora params in ckpt withoud backbone params
        if pet or recover:
            print("<<< loading calora ckpt >>>")
            ckpt = torch.load(calora_path)
            model.load_state_dict(ckpt, strict=False)
            bmt.print_rank("inherit success")
        bmt.synchronize()
        return model

class QloraModel(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            model_name = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            tok_path = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            adapters_name = 'timdettmers/qlora-alpaca-13b'

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
            )
            self.model = PeftModel.from_pretrained(self.model, adapters_name)
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class Nf4Model(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            model_name = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            tok_path = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            adapters_name = 'timdettmers/qlora-alpaca-13b'

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

            self.model.eval()

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class BmquantcaloraModel(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    beam_search: LlamaBeamSearch = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            pretrained = "path/llama-13b"
            quant_config_path = "/data/instruct-eval-main/quantization.json"
            calora_path = ""


            self.tokenizer = LlamaTokenizerBmt.from_pretrained(pretrained)
            self.model = Llama.from_pretrained(pretrained)

            
            ckconfig = bmcook_config.ConfigParser(quant_config_path) 
            BMQuant.quantize(self.model,ckconfig)
            delta_model = LoraModelbmt(
                backbone_model=self.model,
                modified_modules=['project_q', 'project_v', 'project_k', 'attention_out'],
                lora_r=16,
                backend='bmt',
                lora_type='full',
            )
            delta2_model = LoraModelbmt(
                backbone_model=self.model,
                modified_modules=['w_in.w_0', 'w_in.w_1', 'w_out'],
                lora_r = 16, # TODO
                lora_dropout=0.05,
                backend='bmt',
                lora_type='activate',
            )
            delta2_model.log()
            # bmt.load(self.model, calora_path, strict=False)

            self.beam_search = LlamaBeamSearch(
                model=self.model,
                tokenizer=self.tokenizer,
            )

            self.model.eval()

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer([prompt])
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=self.max_output_length,
        #     pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
        #     **kwargs,
        # )
        # from IPython import embed; embed()
        outputs = self.beam_search.generate(
            inputs['input_ids'],
            # attention_mask=attention_mask,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_length=self.max_output_length,
            # stopping_criteria=stopping_criteria,
            # do_sample=False,
        )
        # batch_size, length = inputs.input_ids.shape
        batch_size = 1
        length = len(inputs['input_ids'][0])
        return self.tokenizer.decode(outputs[0][:min(self.max_output_length, length)], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class QloraInheritModel(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            model_name = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            tok_path = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            # adapters_name = 'timdettmers/qlora-alpaca-13b'

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                torch_dtype=torch.float32,
                device_map="auto",
                max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
            )
            # self.model = PeftModel.from_pretrained(self.model, adapters_name)

            # delta_model = LoraModelQlora(
            #     backbone_model=self.model,
            #     modified_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            #     lora_r = 64,
            #     lora_dropout=0.05,
            #     backend='hf',
            #     lora_type='normal',
            # )
            delta_model = LoraModelQlora(
                backbone_model=self.model,
                modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                lora_r=16,
                backend='hf',
                lora_type='normal',
            )
            ckpt = torch.load("path/calora-awq/inherit-old-hf.pt")
            self.model.load_state_dict(ckpt, strict=False)
            delta_model.log()
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

            self.model.eval()

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class AlpacaLoraModel(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            model_name = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            tok_path = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            
            peft_path = '/data/alpaca-lora-13b'

            self.model = PeftModel.from_pretrained(self.model, peft_path)
            print(self.model)


    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B



class QloracaloraModel(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            model_name = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            tok_path = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"
            adapters_name = 'timdettmers/qlora-alpaca-13b'

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                torch_dtype=torch.float32,
                device_map="auto",
                max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path, unk_token ="<s>")

            delta_model = LoraModelQlora(
                 backbone_model=self.model,
                 modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                 lora_r=16,
                 backend='hf',
                 lora_type='full',
            )
            delta2_model = LoraModelQlora(
                 backbone_model=self.model,
                 modified_modules=['gate_proj', 'up_proj', 'down_proj'],
                 lora_r = 16, # TODO
                 lora_dropout=0.05,
                 backend='hf',
                 lora_type='activate',
            )
            # delta2_model = LoraModelQlora(
            #     backbone_model=self.model,
            #     modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            #     lora_r = 64, # TODO
            #     lora_dropout=0.05,
            #     backend='hf',
            #     lora_type='normal',
            # )
            # delta_model = LoraModelQlora(
            #     backbone_model=self.model,
            #     modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            #     lora_r=16,
            #     backend='hf',
            #     lora_type='normal',
            # )
            # delta2_model = LoraModelQlora(
            #     backbone_model=self.model,
            #    modified_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            #    lora_r = 16, # TODO
            #    lora_dropout=0.05,
            #    backend='hf',
            #    lora_type='activate',
            #)

            ckpt = torch.load('/cdgm0705/hyx/lora-thin-2000.pt', map_location=lambda storage, loc: storage.cuda(0))
            # ckpt.pop("model.embed_tokens.weight")
            # ckpt.pop("lm_head.weight")
            self.model.load_state_dict(ckpt, strict=False)
            delta_model.log()

            self.model.eval()

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class PlainModel(EvalModel):
    max_input_length: int = 512
    max_output_length: int = 512
    model_path: str

    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)

            model_name = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"

            tok_path = "/cdgm0705/llama-13b-hf/skyline2006_llama-13b"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

            
            self.model.eval()

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "RWForCausalLM" in str(type(self.model)):
            inputs.pop("token_type_ids")  # Not used by Falcon model

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
    
    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class LlamaModel(SeqToSeqModel):
    use_template: bool = False
    """
    Not officially supported by AutoModelForCausalLM, so we need the specific class
    Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    However, initial MMLU experiments indicate that the template is not useful for few-shot settings
    """

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = LlamaForCausalLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        if self.use_template:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            )
            text = template.format_map(dict(instruction=prompt))
        else:
            text = prompt

        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if "65b" in self.model_path.lower():
            self.max_input_length = 1024
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> Tuple[float, float]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def noop(*args, **kwargs):
    assert args is not None
    assert kwargs is not None


def load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    fused_mlp=True,
    warmup_autotune=True,
):
    config = LlamaConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()

    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]

    quant.make_quant_linear(model, layers, wbits, groupsize)
    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)

    model.seqlen = 2048
    print("Done.")
    return model


class GPTQModel(LlamaModel):
    quantized_path: str
    model: Optional[LlamaForCausalLM]
    tokenizer: Optional[LlamaTokenizer]
    num_bits: int = 4
    group_size: int = 128

    def load(self):
        # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/05781593c818d4dc8adc2d32c975e83d17d2b9a8/llama_inference.py
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not Path(self.quantized_path).exists():
            url = f"https://huggingface.co/{self.model_path}/resolve/main/{self.quantized_path}"
            download_url(url, root=".")

        if self.model is None:
            self.model = load_quant(
                model=self.model_path,
                checkpoint=self.quantized_path,
                wbits=self.num_bits,
                groupsize=self.group_size,
            )
            self.model.to(self.device)

        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            self.test_max_length()

    def test_max_length(self):
        # Detect any OOMs at the beginning
        text = " ".join(["test sentence for max length"] * 1000)
        self.run(text)


class ChatGLMModel(SeqToSeqModel):
    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            ).half()  # FP16 is required for ChatGLM
            self.model.eval()
            self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            **kwargs,
        )
        return response


class RWKVModel(EvalModel):
    tokenizer_path: str = (
        "https://github.com/BlinkDL/ChatRWKV/raw/main/20B_tokenizer.json"
    )
    download_root: str = "."
    model: Optional[rwkv.utils.PIPELINE]

    def download(self, url: str) -> str:
        path = Path(self.download_root, Path(url).name)
        if not path.exists():
            download_url(url, root=self.download_root)
        return str(path)

    def load(self):
        model_path = self.download(self.model_path)
        tokenizer_path = self.download(self.tokenizer_path)

        if self.model is None:
            model = RWKV(model=model_path, strategy="cuda fp16")
            self.model = rwkv.utils.PIPELINE(model, tokenizer_path)

    def run(self, prompt: str, **kwargs) -> str:
        # Adapted from: https://github.com/BlinkDL/ChatRWKV/blob/main/v2/benchmark_more.py
        self.load()
        out_tokens = []
        out_last = 0
        out_str = ""
        occurrence = {}
        state = None
        token = None

        # ctx = f"Bob: {prompt.strip()}\n\nAlice:"
        ctx = prompt  # Special format has lower few-shot performance

        for i in range(self.max_output_length):
            tokens = self.model.encode(ctx) if i == 0 else [token]

            out, state = self.model.model.forward(tokens, state)
            for n in occurrence:
                out[n] -= 0.2 + occurrence[n] * 0.2

            token = self.model.sample_logits(out, temperature=1.0, top_p=0)
            if token == 0:
                break  # exit when 'endoftext'

            out_tokens += [token]
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = self.model.decode(out_tokens[out_last:])
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                # only print when the string is valid utf-8 and not end with \n
                out_str += tmp
                out_last = i + 1

            if "\n\n" in tmp:
                break  # exit when '\n\n'

        return out_str

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.model.encode(text))


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        seq_to_seq=SeqToSeqModel,
        causal=CausalModel,
        llama=LlamaModel,
        chatglm=ChatGLMModel,
        openai=OpenAIModel,
        rwkv=RWKVModel,
        gptq=GPTQModel,
        llama_awq_calora=AwqModel,
        qlora=QloraModel,
        nf4=Nf4Model,
        qlora_inherit=QloraInheritModel,
        qlora_calora=QloracaloraModel,
        alpaca_lora=AlpacaLoraModel,
        bmquant_calora=BmquantcaloraModel,
        plain=PlainModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Write an email about an alpaca that likes flan.",
    model_name: str = "seq_to_seq",
    model_path: str = "google/flan-t5-base",
    **kwargs,
):
    model = select_model(model_name, model_path=model_path, **kwargs)
    print(locals())
    print(model.run(prompt))


"""
p modeling.py test_model --model_name causal --model_path gpt2
p modeling.py test_model --model_name llama --model_path decapoda-research/llama-7b-hf
p modeling.py test_model --model_name llama --model_path chavinlo/alpaca-native
p modeling.py test_model --model_name chatglm --model_path THUDM/chatglm-6b
p modeling.py test_model --model_name llama --model_path TheBloke/koala-7B-HF
p modeling.py test_model --model_name llama --model_path eachadea/vicuna-13b --load_8bit
p modeling.py test_model --model_name causal --model_path togethercomputer/GPT-NeoXT-Chat-Base-20B --load_8bit
p modeling.py test_model --model_name llama --model_path huggyllama/llama-7b --lora_path tloen/alpaca-lora-7b
p modeling.py test_model --model_name seq_to_seq --model_path google/flan-t5-xl --lora_path declare-lab/flan-alpaca-xl-lora
p modeling.py test_model --model_name openai --model_path openai_info.json
p modeling.py test_model --model_name rwkv --model_path https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-7B-v11-Eng99%25-Other1%25-20230427-ctx8192.pth
p modeling.py test_model --model_name causal --model_path mosaicml/mpt-7b-instruct
p modeling.py test_model --model_name gptq --model_path TheBloke/alpaca-lora-65B-GPTQ-4bit --quantized_path alpaca-lora-65B-GPTQ-4bit-128g.safetensors
"""


if __name__ == "__main__":
    Fire()
