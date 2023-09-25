from transformers import AutoTokenizer, LlamaForCausalLM , AutoModelForCausalLM
import random 
import numpy as np
import torch 
from utils import covert_prompt_to_input_ids_with_history
import os 
from flagai.model.file_utils import _get_model_id, _get_checkpoint_path, _get_vocab_path, _get_model_files
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BitsAndBytesConfig,
)
from fastchat.conversation import get_conv_template


def covert_prompt_to_input_ids_with_history(text, history, tokenizer, max_token, convo_template="aquila-chat"):
    # aquila-chat as default
    conv = get_conv_template(convo_template)

    conv.append_message(conv.roles[1], None)
    conv.append_message(conv.roles[0], text)

    example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']

    while(len(history) > 0 and (len(example) < max_token)):
        tmp = history.pop()
        if tmp[0] == 'ASSISTANT':
            conv.append_message(conv.roles[1], tmp[1])
        else:
            conv.append_message(conv.roles[0], tmp[1])
        example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']

    if len(example) >= max_token:
        conv.messages.pop()
    conv.messages = conv.messages[::-1]
    print('model in:', conv.get_prompt())
    example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']

    return example


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class Aquila2Model(LlamaForCausalLM):

    @classmethod
    def from_pretrain(self, model_dir, model_name, **kwargs):
        download_path = os.path.join(model_dir, model_name)
        if os.path.exists(download_path):
            return self.from_pretrained(download_path, **kwargs)


        config_path = os.path.join(download_path, "config.json")
        checkpoint_path = os.path.join(download_path, "pytorch_model.bin")
        from flagai.model.file_utils import _get_model_id
        model_id = _get_model_id(model_name)
        if model_id and model_id != "null":
            model_files = eval(_get_model_files(model_name))
            print("model files:" + str(model_files))
            for file_name in model_files:
                if not file_name.endswith("bin"):
                    _get_vocab_path(download_path, file_name, model_id)

            if os.path.exists(
                    os.path.join(download_path, 'config.json')):
                if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
                    model_parallel_size = int(os.getenv("MODEL_PARALLEL_SIZE"))
                    if model_parallel_size > 1:
                        # if gpus == nums_of_modelhub_models
                        # can load
                        # else need to download the pytorch_model.bin and to recut.
                        model_hub_parallel_size = 0
                        for f in model_files:
                            if "pytorch_model_" in f:
                                model_hub_parallel_size += 1
                else:
                    model_parallel_size = 1

                if "pytorch_model_01.bin" in model_files and model_parallel_size > 1 and model_hub_parallel_size == model_parallel_size:
                    # Only to download the model slices(megatron-lm).
                    for file_to_load in model_files:
                        if "pytorch_model_" in file_to_load:
                            _get_checkpoint_path(download_path, file_to_load,
                                                 model_id)

                elif 'pytorch_model.bin' in model_files:
                    checkpoint_path = _get_checkpoint_path(
                        download_path, 'pytorch_model.bin', model_id)
                else:
                    checkpoint_merge = {}
                    # maybe multi weights files
                    for file_to_load in model_files:
                        if "pytorch_model-0" in file_to_load:
                            _get_checkpoint_path(download_path, file_to_load,
                                                 model_id)
                    #         checkpoint_to_load = torch.load(os.path.join(
                    #             download_path, file_to_load),
                    #                                         map_location="cpu")
                    #         for k, v in checkpoint_to_load.items():
                    #             checkpoint_merge[k] = v
                    # # save all parameters
                    # torch.save(
                    #     checkpoint_merge,
                    #     os.path.join(download_path, "pytorch_model.bin"))


    def predict(self, text, tokenizer=None,
                max_gen_len=200, top_p=0.95,
                seed=1234, topk=100,
                temperature=0.9, 
                sft=True, convo_template = "aquila-chat",
                device = "cuda"):

        vocab = tokenizer.get_vocab()
        #device = device
        id2word = {v:k for k, v in vocab.items()}


        set_random_seed(seed)
        if temperature == 0:
            topk = 1
            temperature = 1.0
        if sft:
            tokens = covert_prompt_to_input_ids_with_history(text, history=[], tokenizer=tokenizer, max_token=2048, convo_template=convo_template)
            tokens = torch.tensor(tokens)[None,].to(device)
        else :
            tokens = tokenizer.encode_plus(text)["input_ids"]
            print(tokenizer.decode(tokens))
            tokens = torch.tensor(tokens)[None,].to(device)
        input_length = len(tokens[0])
        with torch.no_grad():

            # instantiate logits processors
            logits_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(1, eos_token_id=100007),
                ]
            )
            # instantiate logits processors
            logits_warper = LogitsProcessorList(
                [
                    TopPLogitsWarper(top_p),
                    TopKLogitsWarper(topk),
                    TemperatureLogitsWarper(temperature),
                    
                ]
            )

            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=input_length + max_gen_len)])
            out = self.sample(
                                tokens,
                                logits_processor=logits_processor,
                                logits_warper=logits_warper,
                                stopping_criteria=stopping_criteria,
                                return_dict_in_generate=True, 
                                output_scores=True,
                            )

            
            # print(out)
            out_ids = out["sequences"][0][input_length:].cpu().numpy()

            out_scores = out["scores"]

            out_scores = torch.cat(out_scores, dim=0)
            out_scores = torch.nn.functional.softmax(out_scores, dim=-1).cpu().numpy()

            probs = []
            for i in range(len(out_ids)):
                probs.append(float(out_scores[i][out_ids[i]]))

            # print(f"probs is {probs}")

            convert_tokens = []
            for t in out_ids:
                if t == 100006:
                    convert_tokens.append("[CLS]")
                else :
                    convert_tokens.append(id2word.get(t, "[unkonwn_token]"))

            out_text = tokenizer.decode(out_ids.tolist())
            

            out = out_text

        if "###" in out:
            special_index = out.index("###")
            out = out[: special_index]
            token_length = len(tokenizer.encode_plus(out)["input_ids"])
            convert_tokens = convert_tokens[:token_length]
            probs = probs[:token_length]

        if "[UNK]" in out:
            special_index = out.index("[UNK]")
            out = out[:special_index]
            token_length = len(tokenizer.encode_plus(out)["input_ids"])
            convert_tokens = convert_tokens[:token_length]
            probs = probs[:token_length]

        if "</s>" in out:
            special_index = out.index("</s>")
            out = out[: special_index]
            token_length = len(tokenizer.encode_plus(out)["input_ids"])
            convert_tokens = convert_tokens[:token_length]
            probs = probs[:token_length]

        if len(out) > 0 and out[0] == " ":
            out = out[1:]

            convert_tokens = convert_tokens[1:]
            probs = probs[1:]
        return out 
        # return out, convert_tokens, probs
