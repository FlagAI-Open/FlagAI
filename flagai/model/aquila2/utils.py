import random
import numpy as np 
import torch 
from flagai.model.aquila2.conversation import get_conv_template

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)



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