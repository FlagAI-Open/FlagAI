# import jsonlines
import samples_from_conversation as conversation_lib

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    #END_SIGNAL = "" ldwang
    END_SIGNAL = "\n"
    conversation = header
    source["chat_desc"] = header
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": conversation_lib.default_conversation.roles[0],  # human role
        "gpt": conversation_lib.default_conversation.roles[1],  # gpt role
    }
    if "instruction" in source and source["instruction"] is not None and len(source["instruction"]) > 0:
        source["instruction"] = (
            BEGIN_SIGNAL
            + "System: "
            + source["instruction"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += source["instruction"]
    for sentence in source["conversations"]:
        sentence_from = sentence["from"].lower()
        if sentence_from == 'human':
            END_SIGNAL2 = ''
        else:
            END_SIGNAL2 = END_SIGNAL
        sentence["value"] = (
            BEGIN_SIGNAL
            + roles.get(sentence_from, unknown_role)
            + ": "
            + sentence["value"]
            + END_SIGNAL2
        )
        if get_conversation:
            conversation += sentence["value"]
    return conversation

header = "A chat between a curious user and an artificial intelligence assistant. " \
         "The assistant gives helpful, detailed, and polite answers to the user's questions. \n"
         #"The assistant gives helpful, detailed, and polite answers to the user's questions. " ldwang

input_file = 'conversations/sample_data_v0.6_5w_0420.jsonl'
output_file = 'conversations/sample_data_v0.6_5w_0420_dataset.jsonl'
input_file = '/share/project/ldwang/sft/datasets/conversations/merge_chat_clean.jsonl'
output_file = '/share/project/ldwang/sft/datasets/conversations/merge_chat_clean_convo_dataset.jsonl'
input_file = 'conversation_demo.jsonl'
output_file = 'conversation_dataset_demo.jsonl'
# fo = jsonlines.open(output_file, mode='w')
# with jsonlines.open(input_file) as reader:
#     for obj in reader:
#         conversation = _add_speaker_and_signal(header, obj, get_conversation=True)
#         #print(conversation)
#         fo.write(obj)


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

def convert_step1_to_step2(obj):

    conversation = _add_speaker_and_signal(header, obj, get_conversation=True)

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


def get_input_ids(conversations, tokenizer):
    chat_desc = conversations['chat_desc']
    instruction = conversations['instruction']
    conversations = conversations['conversations']
    
    # chat_desc
    example = tokenizer.encode_plus(f"{chat_desc}", None, max_length=None)['input_ids']
    EOS_TOKEN = example[-1]
    example = example[:-1] # remove eos
    # instruction
    if instruction != "":
        instruction = tokenizer.encode_plus(f"{instruction}", None, max_length=None)['input_ids']
        instruction = instruction[1:-1] # remove bos & eos
        example += instruction


    for conversation in conversations:
        role = conversation['from']
        content = conversation['value']
        content = tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
        content = content[1:-1] # remove bos & eos
        example += content
      
    return example


def covert_prompt_to_input_ids(text, tokenizer):
    #obj = pack_obj(text)
    # convert_step1_to_step2(obj)
    # print(obj)
    # delete_last_bot_end_singal(obj)
    # # print(obj)
    # return get_input_ids(obj, tokenizer)

    from conversation_yemin import default_conversation

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)

    #print(conv.get_prompt())
    example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']
    example = example[1:-1]

    return example

def covert_prompt_to_input_ids_new(text, tokenizer):
    #obj = pack_obj(text)
    #convert_step1_to_step2(obj)
    # print(obj)
    #delete_last_bot_end_singal(obj)
    # # print(obj)
    #return get_input_ids(obj, tokenizer)

    from conversation_yemin import default_conversation

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], text)
    #conv.append_message(conv.roles[1], None)

    inputs = dict()
    inputs['prompt'] = conv.get_prompt()
    print(f"prompt {inputs}")
    example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']
    example = example[1:-1]

    return example



if __name__ == "__main__":
    from flagai.data.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_pretrained("llama-30b-en", 
                                      cache_dir="./gpt2_new_100k_newline")


    out = covert_prompt_to_input_ids("你好，世界", tokenizer=tokenizer)
    print(tokenizer.decode(out))

    print('\n')

    out = covert_prompt_to_input_ids_new("你好，世界", tokenizer=tokenizer)
    print(tokenizer.decode(out))

