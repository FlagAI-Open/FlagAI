# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
import gc
gc.collect()
torch.cuda.empty_cache()

from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.env_args import EnvArgs
from flagai.env_trainer_v1 import EnvTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=100,
    log_interval=10,
    eval_interval=5000,
    num_gpus=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_aquila",
    checkpoint_activations=False,
    save_interval=5000,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()
#env_args.wandb = False

# overwrite
if env_args.yaml_config:
    import yaml
    file_data = open(env_args.yaml_config, 'r', encoding="utf-8").read()
    data = yaml.load_all(file_data)
    delattr(env_args, 'yaml_config')
    arg_dict = env_args.__dict__
    for subdata in data:
        for key, value in subdata.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
trainer = EnvTrainer(env_args)

# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

print(f"Trainer effective env_args={env_args} local_rank={trainer.local_rank}", flush=True)

#checkpoints = "/share/project/ldwang/sft/state_dict/"
checkpoints = env_args.pre_load_dir
# model_name = env_args.model_name

# checkpoints = "/data/yzd/FlagAI/examples/aquila/checkpoints_in/"
model_name = env_args.model_name
# model_name = "aquila-7b"
env_args.enable_sft_conversations_dataset_v3 = True


print('*'*20, "model_name", model_name, flush=True)

'''
auto_loader = AutoLoader(
    "lm",
    model_name=model_name,
    model_dir=checkpoints,
    only_download_config=True,
)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
print('*'*20, "model", model)
trainer.pre_train(model)
print('*'*20, "model", model)

'''

cache_dir = os.path.join(checkpoints, model_name)
print('*'*20, "cache_dir", cache_dir)
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print('*'*20, "tokenizer", tokenizer)

# avoid sync loading models in case of Mem OOM
if env_args.bmt_async_load:
    import time
    time.sleep(10*60*(trainer.local_rank%4))


config_file = os.path.join(cache_dir, 'config.json')
from flagai.model.aquila_model import AQUILAModel
model = AQUILAModel.init_from_json(config_file=config_file)
print('*'*20, "model", model)

## bmt_pre_load
checkpoint_path = os.path.join(cache_dir, "pytorch_model.bin")
if env_args.bmt_pre_load:
    model.load_weights(checkpoint_path)

trainer.pre_train(model)

print('*'*20, "model", model, flush=True)


if env_args.enable_sft_conversations_dataset_v3:
    assert env_args.enable_sft_dataset_dir is not None and \
            env_args.enable_sft_dataset_file is not None

    cur_dir = env_args.enable_sft_dataset_dir
    jsonl_data = os.path.join(cur_dir, env_args.enable_sft_dataset_file)
    max_seq_len = 2048

    import jsonlines
    import numpy as np
    def read_file():
        conversations = []
        with jsonlines.open(jsonl_data) as reader:
            for line in reader:
                conversations.append(line)
        return conversations

    from examples.aquila import cyg_conversation as conversation_lib
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": conversation_lib.default_conversation.roles[0],  # human role
        "gpt": conversation_lib.default_conversation.roles[1],  # gpt role
    }

    def _add_speaker_and_signal(header, source, get_conversation=True):
        conversation = header

        if "instruction" in source and source["instruction"] is not None and len(source["instruction"]) > 0:
            source["instruction"] = (
                BEGIN_SIGNAL
                + conversation_lib.default_conversation.roles[2]
                + ": "
                + source["instruction"]
                + END_SIGNAL
            )
            if get_conversation:
                conversation += source["instruction"]
        for sentence in source["conversations"]:
            sentence_from = sentence["from"].lower()
            sentence["value"] = (
                BEGIN_SIGNAL
                + roles.get(sentence_from, unknown_role)
                + ": "
                + sentence["value"]
                + END_SIGNAL
            )
            if get_conversation:
                conversation += sentence["value"]
        return conversation

    class ConversationDatasetV3(Dataset):
        def __init__(self, conversations, tokenizer, maxlen=512):
            super(ConversationDatasetV3, self).__init__()
            self.conversations = conversations
            self.tokenizer = tokenizer
            self.maxlen = maxlen

        def __getitem__(self, i):
            header = f"{conversation_lib.default_conversation.system}\n\n"
            source = self.conversations[i]
            _add_speaker_and_signal(header, source)

            source["chat_desc"] = header
            chat_desc = source['chat_desc']
            instruction = source['instruction']
            conversations = source['conversations']
            
            # chat_desc
            example = self.tokenizer.encode_plus(f"{chat_desc}", None, max_length=None)['input_ids']
            EOS_TOKEN = example[-1]
            example = example[:-1] # remove eos
            # instruction
            instruction = self.tokenizer.encode_plus(f"{instruction}", None, max_length=None)['input_ids']
            instruction = instruction[1:-1] # remove bos & eos
            example += instruction

            import copy
            labels = copy.deepcopy(example)

            for conversation in conversations:
                role = conversation['from']
                content = conversation['value']

                if role == 'gpt':
                    prefix_gpt = BEGIN_SIGNAL + roles.get(role, unknown_role) + ": "
                    content_gpt = content[len(prefix_gpt):]

                    prefix_gpt = self.tokenizer.encode_plus(f"{prefix_gpt}", None, max_length=None)['input_ids']
                    prefix_gpt = prefix_gpt[1:-1] # remove bos & eos
                    example += prefix_gpt
                    role_labels = [env_args.IGNORE_INDEX] * len(prefix_gpt)

                    content_gpt = self.tokenizer.encode_plus(f"{content_gpt}", None, max_length=None)['input_ids']
                    content_gpt = content_gpt[1:-1] # remove bos & eos
                    example += content_gpt
                    role_labels += copy.deepcopy(content_gpt)
                else:
                    content = self.tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
                    content = content[1:-1] # remove bos & eos
                    example += content
                    # masking
                    role_labels = [env_args.IGNORE_INDEX] * len(content)
                labels += role_labels

            example.append(EOS_TOKEN)
            labels.append(EOS_TOKEN)
            assert len(example) == len(labels)

            ## maxlen
            example = example[:self.maxlen]
            labels = labels[:self.maxlen]

            output = {
                "input_ids": example,
                "labels": labels,
            }
            return output

        def __len__(self):
            return len(self.conversations)

        @staticmethod
        def collate_fn(batch):
            def padding(indice, max_length, pad_idx=0):
                pad_indice = [
                    item + [pad_idx] * max(0, max_length - len(item)) for item in indice
                ]
                return torch.tensor(pad_indice)

            input_ids = [data["input_ids"] for data in batch]
            labels = [data["labels"] for data in batch]
            max_length = max_seq_len
            input_ids = padding(input_ids, max_length)[:,:max_length]
            labels = padding(labels, max_length, pad_idx=env_args.IGNORE_INDEX)[:,:max_length]

            data = {
                "input_ids": input_ids,
                "labels": labels
            }
            return data

    conversations = read_file()
    data_len = len(conversations)
    #train_size = int(data_len * 0.95)
    train_size = data_len
    train_conversations = conversations[:train_size]

    train_dataset = ConversationDatasetV3(train_conversations,
                                            tokenizer=tokenizer,
                                            maxlen=max_seq_len)
    #print(f"train_dataset \n {train_dataset[0]}")

    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=ConversationDatasetV3.collate_fn,
        optimizer=None,
        rank_split=False)
elif env_args.enable_sft_conversations_dataset_v3:
    assert env_args.enable_sft_dataset_dir is not None and \
           env_args.enable_sft_dataset_file is not None

    cur_dir = env_args.enable_sft_dataset_dir
    jsonl_data = os.path.join(cur_dir, env_args.enable_sft_dataset_file)
    max_seq_len = 2048

    import jsonlines
    import numpy as np
    def read_file():
        conversations = []
        with jsonlines.open(jsonl_data) as reader:
            for line in reader:
                conversations.append(line)
        return conversations

    from examples.gpt3_pretrain.llama import ym_conversation as conversation_lib
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": conversation_lib.default_conversation.roles[0],  # human role
        "gpt": conversation_lib.default_conversation.roles[1],  # gpt role
    }

    def _add_speaker_and_signal(header, source, get_conversation=True):
        conversation = header

        if "instruction" in source and source["instruction"] is not None and len(source["instruction"]) > 0:
            source["instruction"] = (
                BEGIN_SIGNAL
                + conversation_lib.default_conversation.roles[2]
                + ": "
                + source["instruction"]
                + END_SIGNAL
            )
            if get_conversation:
                conversation += source["instruction"]
        for sentence in source["conversations"]:
            sentence_from = sentence["from"].lower()
            sentence["value"] = (
                BEGIN_SIGNAL
                + roles.get(sentence_from, unknown_role)
                + ": "
                + sentence["value"]
                + END_SIGNAL
            )
            if get_conversation:
                conversation += sentence["value"]
        return conversation
    
    class ConversationDatasetV3(Dataset):
        def __init__(self, conversations, tokenizer, maxlen=512):
            super(ConversationDatasetV3, self).__init__()
            self.conversations = conversations
            self.tokenizer = tokenizer
            self.maxlen = maxlen
    
        def __getitem__(self, i):
            header = f"{conversation_lib.default_conversation.system}\n\n"
            source = self.conversations[i]
            _add_speaker_and_signal(header, source)

            source["chat_desc"] = header
            chat_desc = source['chat_desc']
            instruction = source['instruction']
            conversations = source['conversations']
            
            # chat_desc
            example = self.tokenizer.encode_plus(f"{chat_desc}", None, max_length=None)['input_ids']
            EOS_TOKEN = example[-1]
            example = example[:-1] # remove eos
            # instruction
            instruction = self.tokenizer.encode_plus(f"{instruction}", None, max_length=None)['input_ids']
            instruction = instruction[1:-1] # remove bos & eos
            example += instruction

            import copy
            labels = copy.deepcopy(example)

            for conversation in conversations:
                role = conversation['from']
                content = conversation['value']

                if role == 'gpt':
                    prefix_gpt = BEGIN_SIGNAL + roles.get(role, unknown_role) + ": "
                    content_gpt = content[len(prefix_gpt):]

                    prefix_gpt = self.tokenizer.encode_plus(f"{prefix_gpt}", None, max_length=None)['input_ids']
                    prefix_gpt = prefix_gpt[1:-1] # remove bos & eos
                    example += prefix_gpt
                    role_labels = [env_args.IGNORE_INDEX] * len(prefix_gpt)

                    content_gpt = self.tokenizer.encode_plus(f"{content_gpt}", None, max_length=None)['input_ids']
                    content_gpt = content_gpt[1:-1] # remove bos & eos
                    example += content_gpt
                    role_labels += copy.deepcopy(content_gpt)
                else:
                    content = self.tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
                    content = content[1:-1] # remove bos & eos
                    example += content
                    # masking
                    role_labels = [env_args.IGNORE_INDEX] * len(content)
                labels += role_labels

            example.append(EOS_TOKEN)
            labels.append(EOS_TOKEN)
            assert len(example) == len(labels)

            ## maxlen
            example = example[:self.maxlen]
            labels = labels[:self.maxlen]

            output = {
                "input_ids": example,
                "labels": labels,
            }
            return output
    
        def __len__(self):
            return len(self.conversations)
    
        @staticmethod
        def collate_fn(batch):
            def padding(indice, max_length, pad_idx=0):
                pad_indice = [
                    item + [pad_idx] * max(0, max_length - len(item)) for item in indice
                ]
                return torch.tensor(pad_indice)
    
            input_ids = [data["input_ids"] for data in batch]
            labels = [data["labels"] for data in batch]
            max_length = max_seq_len
            input_ids = padding(input_ids, max_length)[:,:max_length]
            labels = padding(labels, max_length, pad_idx=env_args.IGNORE_INDEX)[:,:max_length]
    
            data = {
                "input_ids": input_ids,
                "labels": labels
            }
            return data
    
    conversations = read_file()
    data_len = len(conversations)
    #train_size = int(data_len * 0.95)
    train_size = data_len
    train_conversations = conversations[:train_size]

    train_dataset = ConversationDatasetV3(train_conversations,
                                          tokenizer=tokenizer,
                                          maxlen=max_seq_len)
    #print(f"train_dataset \n {train_dataset[0]}")

    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=ConversationDatasetV3.collate_fn,
        optimizer=None,
        rank_split=False)
