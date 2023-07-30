import os
import uvicorn, json
from asgiref.sync import sync_to_async
import os
import random 
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
from cyg_conversation import covert_prompt_to_input_ids_with_history
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

model_name = "aquilachat-7b-huggingface"
server_port = 5050

device = "cuda:0"

def load():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/AquilaChat-7B")
    model = AutoModelForCausalLM.from_pretrained("BAAI/AquilaChat-7B")
    model.half()
    model.eval()
    model.to("cuda:0")
    return model, tokenizer 

model, tokenizer = load()
vocab = tokenizer.get_vocab()
id2word = {v:k for k, v in vocab.items()}


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def predict(text,
            max_gen_len=200, top_p=0.95,
            seed=1234, topk=100,
            temperature=0.9, 
            sft=True):
    set_random_seed(seed)
    if sft:
        tokens = covert_prompt_to_input_ids_with_history(text, history=[], tokenizer=tokenizer, max_token=2048)
        tokens = torch.tensor(tokens)[None,].to(device)

    else :
        tokens = tokenizer.encode_plus(text)["input_ids"][:-1]
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

        out = model.sample(
                            tokens,
                            logits_processor=logits_processor,
                            logits_warper=logits_warper,
                            stopping_criteria=stopping_criteria,
                            return_dict_in_generate=True, 
                            output_scores=True,
                        )

        
        # print(out)
        out_ids = out["sequences"][0][input_length+1: ].cpu().numpy()

        out_scores = out["scores"]

        out_scores = torch.cat(out_scores, dim=0)[1:]
        out_scores = torch.nn.functional.softmax(out_scores, dim=-1).cpu().numpy()

        probs = []
        for i in range(len(out_ids)):
            probs.append(float(out_scores[i][out_ids[i]]))

        print(f"probs is {probs}")

        convert_tokens = []
        for t in out_ids:
            if t == 100006:
                convert_tokens.append("[CLS]")
            else :
                convert_tokens.append(id2word.get(t, "[unkonwn_token]"))

        out_text = tokenizer.decode(out_ids.tolist())
        print(out_text)

        out = out_text

    if "###" in out:
        special_index = out.index("###")
        out = out[: special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"][1:-1])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if "[UNK]" in out:
        special_index = out.index("[UNK]")
        out = out[:special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"][1:-1])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if "</s>" in out:
        special_index = out.index("</s>")
        out = out[: special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"][1:-1])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if len(out) > 0 and out[0] == " ":
        out = out[1:]

        convert_tokens = convert_tokens[1:]
        probs = probs[1:]

    return out, convert_tokens, probs

def init_flask():
    from fastapi import FastAPI, Request

    app = FastAPI()

    @app.post("/func")
    async def get_generate_h(request: Request):
        json_post_raw = await request.json()
        config = json.loads(json_post_raw)

        text = config["prompt"]
        topp = config.get("top_p", 0.95)
        max_length = config.get("max_new_tokens", 256)
        topk = config.get("top_k_per_token", 1000)
        temperature = config.get("temperature", 0.9)
        sft = config.get("sft", False)
        seed = config.get("seed", 1234)

        print(f"sft is {sft}")
        out, tokens, probs = await sync_to_async(predict)(text,
                                           max_gen_len=max_length, top_p=topp,
                                           topk=topk,
                                           temperature=temperature, sft=sft,
                                           seed=seed)

        result = {
            "completions": [{
                "text": out,
                "tokens": tokens,
                "logprobs": probs,
                "top_logprobs_dicts": [{k: v} for k, v in zip(tokens, probs)],
            }],
            "input_length": len(config["prompt"]),
            "model_info":model_name}

        return result

    return app 

app = init_flask()

uvicorn.run(app, host='0.0.0.0', port=server_port, workers=1)
