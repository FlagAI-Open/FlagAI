import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.aquila import aquila_generate
from flagai.data.tokenizer import Tokenizer
import bminf

state_dict = "/data2/yzd/checkpoints/converted_models_ldwang"
model_name = 'aquilachat-7b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()
cache_dir = os.path.join(state_dict, model_name)

model.eval()
model.half()

with torch.cuda.device(0):
    model = bminf.wrapper(model, quantization=False, memory_limit=2 << 30)

predictor = Predictor(model, tokenizer)

texts = [
        "北京为什么是中国的首都？",
        "1+1=",
        "为什么湘菜那么甜？",
        "东三省和海南岛的区别？",
        ]
## 
def pack_obj(text):
    obj = dict()
    obj['id'] = 'demo'

    obj['conversations'] = []
    human = dict()
    human['from'] = 'human'
    human['value'] = text
    obj['conversations'].append(human)
    # dummy bot
    bot = dict()
    bot['from'] = 'gpt'
    bot['value'] = ''
    obj['conversations'].append(bot)

    obj['instruction'] = ''

    return obj

def delete_last_bot_end_singal(convo_obj):
    conversations = convo_obj['conversations']
    assert len(conversations) > 0 and len(conversations) % 2 == 0
    assert conversations[0]['from'] == 'human'

    last_bot = conversations[len(conversations)-1]
    assert last_bot['from'] == 'gpt'

    ## from _add_speaker_and_signal
    END_SIGNAL = "\n"
    len_end_singal = len(END_SIGNAL)
    len_last_bot_value = len(last_bot['value'])
    last_bot['value'] = last_bot['value'][:len_last_bot_value-len_end_singal]
    return

def convo_tokenize(convo_obj, tokenizer):
    chat_desc = convo_obj['chat_desc']
    instruction = convo_obj['instruction']
    conversations = convo_obj['conversations']
            
    # chat_desc
    example = tokenizer.encode_plus(f"{chat_desc}", None, max_length=None)['input_ids']
    EOS_TOKEN = example[-1]
    example = example[:-1] # remove eos
    # instruction
    instruction = tokenizer.encode_plus(f"{instruction}", None, max_length=None)['input_ids']
    instruction = instruction[1:-1] # remove bos & eos
    example += instruction

    for conversation in conversations:
        role = conversation['from']
        content = conversation['value']
        print(f"role {role}, raw content {content}")
        content = tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
        content = content[1:-1] # remove bos & eos
        print(f"role {role}, content {content}")
        example += content
    return example

for text in texts:
    print('-'*80)
    print(f"text is {text}")

    from cyg_conversation import default_conversation

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)

    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
    tokens = tokens[1:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer, model, [text], max_gen_len:=200, top_p=0.95, prompts_tokens=[tokens])
        print(f"pred is {out}")