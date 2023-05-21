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

#torch.autograd.set_detect_anomaly(True)

from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_datasets
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_weighted_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    experiment_name="llama",
    model_name="llama-7b-en",
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
    save_dir="checkpoints_llama",
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
model_name = env_args.model_name
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
#print('*'*20, "cache_dir", cache_dir)
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#print('*'*20, "tokenizer", tokenizer)

#config_file = cache_dir + "/config.json"
config_file = os.path.join(cache_dir, 'config.json')
# avoid sync loading models in case of Mem OOM
if env_args.bmt_async_load:
    import time
    time.sleep(10*60*(trainer.local_rank%2))

from flagai.model.llama_model import LLAMAModel
model = LLAMAModel.init_from_json(config_file=config_file)
#print('*'*20, "model", model)

if env_args.bmt_pre_load:
    checkpoint_path = os.path.join(cache_dir, "pytorch_model.bin")
    model.load_weights(checkpoint_path)

trainer.pre_train(model)
print('*'*20, "model", model, flush=True)

## conversations_dataset
if env_args.enable_sft_conversations_dataset_v2:
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
    def _add_speaker_and_signal(header, source, get_conversation=True):

        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        conversation = header
        unknown_role = "unknown"  # use default unknown role
        roles = {
            "human": conversation_lib.default_conversation.roles[0],  # human role
            "gpt": conversation_lib.default_conversation.roles[1],  # gpt role
        }
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
    
    class ConversationDatasetV2(Dataset):
        def __init__(self, conversations, tokenizer, maxlen=512):
            super(ConversationDatasetV2, self).__init__()
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
                content = self.tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
                content = content[1:-1] # remove bos & eos
                example += content
                if role == 'gpt':
                    role_labels = copy.deepcopy(content)
                else:
                    # masking
                    role_labels = [env_args.IGNORE_INDEX] * len(content)
                labels += role_labels

            example.append(EOS_TOKEN)
            labels.append(EOS_TOKEN)

            ## delete bos & eos
            example = example[1:-1]
            labels = labels[1:-1]

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

    train_dataset = ConversationDatasetV2(train_conversations,
                                          tokenizer=tokenizer,
                                          maxlen=max_seq_len)
    #print(f"train_dataset \n {train_dataset[0]}")

    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=ConversationDatasetV2.collate_fn,
        optimizer=None,
        rank_split=False)

elif env_args.enable_sft_conversations_dataset:
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
                if 'chat_desc' not in line or 'instruction' not in line or 'conversations' not in line:
                    continue
                obj = dict()
                obj['chat_desc'] = line['chat_desc']
                obj['conversations'] = line['conversations']
                obj['instruction'] = line['instruction']
                conversations.append(obj)
        return conversations
    
    class ConversationDataset(Dataset):
        def __init__(self, conversations, tokenizer, maxlen=512):
            super(ConversationDataset, self).__init__()
            self.conversations = conversations
            self.tokenizer = tokenizer
            self.maxlen = maxlen
    
        def __getitem__(self, i):
            chat_desc = self.conversations[i]['chat_desc']
            instruction = self.conversations[i]['instruction']
            conversations = self.conversations[i]['conversations']
            
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
                content = self.tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
                content = content[1:-1] # remove bos & eos
                example += content
                if role == 'gpt':
                    role_labels = copy.deepcopy(content)
                else:
                    # masking
                    role_labels = [env_args.IGNORE_INDEX] * len(content)
                labels += role_labels

            example.append(EOS_TOKEN)
            labels.append(EOS_TOKEN)

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

    train_dataset = ConversationDataset(train_conversations,
                                       tokenizer=tokenizer,
                                       maxlen=max_seq_len)

    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=ConversationDataset.collate_fn,
        optimizer=None,
        rank_split=False)

    
elif env_args.enable_sft_dataset:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ## v0.5
    json_data = os.path.join(cur_dir, 'data/sample_data_10w_0416.json')
    max_seq_len = 2048
    
    import json
    import numpy as np
    def read_file():
        src = []
        tgt = []
        with open(json_data, 'r', encoding='utf-8') as f:
            lines = json.load(f)
            for line in lines:
                if 'response' not in line or 'prompt' not in line:
                    continue
                src.append(line['prompt'].strip('\n'))
                tgt.append(line['response'].strip('\n'))

        return src, tgt

    class InstructionDataset(Dataset):
        def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
            super(InstructionDataset, self).__init__()
            self.sents_src = sents_src
            self.sents_tgt = sents_tgt
            self.tokenizer = tokenizer
            self.maxlen = maxlen
    
        def __getitem__(self, i):
            src = self.sents_src[i]
            #[:self.maxlen]
            tgt = self.sents_tgt[i]
            
            ## Based on different tokenizers
            prompt = self.tokenizer.encode_plus(src, None, max_length=None)['input_ids']
            ## remove eos
            prompt = prompt[:-1]
            ## maxlen
            prompt = prompt[:self.maxlen]
            ## bos+ids+eos
            example = self.tokenizer.encode_plus(f"{src}{tgt}", None, max_length=None)['input_ids']
            ## maxlen
            example = example[:self.maxlen]

            prompt = torch.tensor(prompt, dtype=torch.int64)
            example = torch.tensor(example, dtype=torch.int64)

            import copy
            labels = copy.deepcopy(example)
            prompt_len = len(prompt)
            labels[:prompt_len] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = 0
    
            output = {
                "input_ids": example.tolist(),
                "labels": labels.tolist(),
            }
            return output
    
        def __len__(self):
            return len(self.sents_src)
    
        @staticmethod
        def collate_fn(batch):
            def padding(indice, max_length, pad_idx=0):
                pad_indice = [
                    item + [pad_idx] * max(0, max_length - len(item)) for item in indice
                ]
                return torch.tensor(pad_indice)
    
            input_ids = [data["input_ids"] for data in batch]
            labels = [data["labels"] for data in batch]
            #max_length = max([len(t) for t in input_ids])
            #max_length_labels = max([len(t) for t in labels])
            #assert max_length == max_length_labels
            max_length = max_seq_len
            input_ids = padding(input_ids, max_length)[:,:max_length]
            labels = padding(labels, max_length)[:,:max_length]
    
            data = {
                "input_ids": input_ids,
                "labels": labels
            }
            return data
    
    sents_src, sents_tgt = read_file()
    data_len = len(sents_tgt)
    #train_size = int(data_len * 0.95)
    train_size = data_len
    train_src = sents_src[:train_size]
    train_tgt = sents_tgt[:train_size]

    train_dataset = InstructionDataset(train_src,
                                       train_tgt,
                                       tokenizer=tokenizer,
                                       maxlen=max_seq_len)

    '''
    valid_src = sents_src[train_size:]
    valid_tgt = sents_tgt[train_size:]
    valid_dataset = InstructionDataset(valid_src,
                                       valid_tgt,
                                       tokenizer=tokenizer,
                                       maxlen=max_seq_len)
    '''

    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=InstructionDataset.collate_fn,
        optimizer=None,
        rank_split=False)

elif env_args.enable_sft_dataset_jsonl:
    ## TODO
    if env_args.enable_sft_dataset_dir:
        cur_dir = env_args.enable_sft_dataset_dir
        #jsonl_data = os.path.join(cur_dir, 'merge_chat_clean_dataset.jsonl')
        jsonl_data = os.path.join(cur_dir, env_args.enable_sft_dataset_file)
    else:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        jsonl_data = os.path.join(cur_dir, 'data/sample_data_10w_0416.json')
    max_seq_len = 2048
    
    import jsonlines
    import numpy as np
    def read_file():
        src = []
        tgt = []
        with jsonlines.open(jsonl_data) as reader:
            for line in reader:
                if 'response' not in line or 'prompt' not in line:
                    continue
                src.append(line['prompt'].strip('\n'))
                tgt.append(line['response'].strip('\n'))

        return src, tgt

    
    class InstructionDataset(Dataset):
        def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
            super(InstructionDataset, self).__init__()
            self.sents_src = sents_src
            self.sents_tgt = sents_tgt
            self.tokenizer = tokenizer
            self.maxlen = maxlen
    
        def __getitem__(self, i):
            src = self.sents_src[i]
            #[:self.maxlen]
            tgt = self.sents_tgt[i]
            
            ## Based on different tokenizers
            prompt = self.tokenizer.encode_plus(src, None, max_length=None)['input_ids']
            ## remove eos
            prompt = prompt[:-1]
            ## maxlen
            prompt = prompt[:self.maxlen]
            ## bos+ids+eos
            example = self.tokenizer.encode_plus(f"{src}{tgt}", None, max_length=None)['input_ids']
            ## maxlen
            example = example[:self.maxlen]

            prompt = torch.tensor(prompt, dtype=torch.int64)
            example = torch.tensor(example, dtype=torch.int64)

            import copy
            labels = copy.deepcopy(example)
            prompt_len = len(prompt)
            labels[:prompt_len] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = 0
    
            output = {
                "input_ids": example.tolist(),
                "labels": labels.tolist(),
            }
            return output
    
        def __len__(self):
            return len(self.sents_src)
    
        @staticmethod
        def collate_fn(batch):
            def padding(indice, max_length, pad_idx=0):
                pad_indice = [
                    item + [pad_idx] * max(0, max_length - len(item)) for item in indice
                ]
                return torch.tensor(pad_indice)
    
            input_ids = [data["input_ids"] for data in batch]
            labels = [data["labels"] for data in batch]
            #max_length = max([len(t) for t in input_ids])
            #max_length_labels = max([len(t) for t in labels])
            #assert max_length == max_length_labels
            max_length = max_seq_len
            input_ids = padding(input_ids, max_length)[:,:max_length]
            labels = padding(labels, max_length)[:,:max_length]
    
            data = {
                "input_ids": input_ids,
                "labels": labels
            }
            return data
    
    sents_src, sents_tgt = read_file()
    data_len = len(sents_tgt)
    #train_size = int(data_len * 0.95)
    train_size = data_len
    train_src = sents_src[:train_size]
    train_tgt = sents_tgt[:train_size]

    train_dataset = InstructionDataset(train_src,
                                       train_tgt,
                                       tokenizer=tokenizer,
                                       maxlen=max_seq_len)

    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=InstructionDataset.collate_fn,
        optimizer=None,
        rank_split=False)

elif env_args.enable_sft_dataset_text:
    if env_args.enable_sft_dataset_dir:
        data_dir = env_args.enable_sft_dataset_dir
    else:
        ## v0.6
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(cur_dir, 'data/sample_data_v0.6_5w_0420')
    max_seq_len = 2048
    src_dir = os.path.join(data_dir, 'train.source')
    tgt_dir = os.path.join(data_dir, 'train.target')

    def read_file():
        src = []
        tgt = []
    
        # text = text.replace('\\n', '\n')
        with open(src_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                src.append(line.strip('\n').replace('\\n', '\n'))
    
        with open(tgt_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tgt.append(line.strip('\n').replace('\\n', '\n'))
    
        assert len(src)==len(tgt), "Samples should be same."
        return src, tgt
    
    class InstructionDataset(Dataset):
        def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=512):
            super(InstructionDataset, self).__init__()
            self.sents_src = sents_src
            self.sents_tgt = sents_tgt
            self.tokenizer = tokenizer
            self.maxlen = maxlen
    
        def __getitem__(self, i):
            src = self.sents_src[i]
            #[:self.maxlen]
            tgt = self.sents_tgt[i]
            
            ## Based on different tokenizers
            prompt = self.tokenizer.encode_plus(src, None, max_length=None)['input_ids']
            ## remove eos
            prompt = prompt[:-1]
            ## maxlen
            prompt = prompt[:self.maxlen]
            ## bos+ids+eos
            example = self.tokenizer.encode_plus(f"{src}{tgt}", None, max_length=None)['input_ids']
            ## maxlen
            example = example[:self.maxlen]

            prompt = torch.tensor(prompt, dtype=torch.int64)
            example = torch.tensor(example, dtype=torch.int64)

            import copy
            labels = copy.deepcopy(example)
            prompt_len = len(prompt)
            labels[:prompt_len] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = 0
    
            output = {
                "input_ids": example.tolist(),
                "labels": labels.tolist(),
            }
            return output
    
        def __len__(self):
            return len(self.sents_src)
    
        @staticmethod
        def collate_fn(batch):
            def padding(indice, max_length, pad_idx=0):
                pad_indice = [
                    item + [pad_idx] * max(0, max_length - len(item)) for item in indice
                ]
                return torch.tensor(pad_indice)
    
            input_ids = [data["input_ids"] for data in batch]
            labels = [data["labels"] for data in batch]
            #max_length = max([len(t) for t in input_ids])
            #max_length_labels = max([len(t) for t in labels])
            #assert max_length == max_length_labels
            max_length = max_seq_len
            input_ids = padding(input_ids, max_length)[:,:max_length]
            labels = padding(labels, max_length)[:,:max_length]
    
            data = {
                "input_ids": input_ids,
                "labels": labels
            }
            return data
    
    sents_src, sents_tgt = read_file()
    data_len = len(sents_tgt)
    #train_size = int(data_len * 0.95)
    train_size = data_len
    train_src = sents_src[:train_size]
    train_tgt = sents_tgt[:train_size]

    train_dataset = InstructionDataset(train_src,
                                       train_tgt,
                                       tokenizer=tokenizer,
                                       maxlen=max_seq_len)

    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=InstructionDataset.collate_fn,
        optimizer=None,
        rank_split=False)

elif env_args.enable_weighted_dataset_v2:
    ## 1: weight01, prefix01, weight02, prefix02, ...
    data_prefix = [
        1.296091,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_9_dedup_wudao_text_document',
        144.325674,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_9_part_merged_text_document',
        53.498074,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pile-cc_text_document',
        23.575721,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-openwebtext2_text_document',
    
        14.718128,
        '/data/indexed_dataset/batch1_tok100k_sep/code_dedup-md5-pile-github_text_document',
        8.878174,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code_text_document',
        3.439587,
        '/data/indexed_dataset/batch1_tok100k_sep/code_newcode1_text_document',
        2.533595,
        '/data/indexed_dataset/batch1_tok100k_sep/code_newcode2_text_document',
        9.410141,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code-cpp_text_document',
        5.965614,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code-java_text_document',
    
        22.442690,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_baike_text_document',
        10.276255,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-wikipedia_en_text_document',
    
        6.821143,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_ebook_merge_maxlen_text_document',
        4.057581,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-gutenberg_pg-19_text_document',
        2.266030,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-bookcorpus2_text_document',
        37.479110,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-books3_text_document',
        20.044762,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-arxiv_text_document',
        4.826957,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pubmed_abstracts_text_document',
    
        7.514409,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_zhihu_text_document',
        19.639909,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-stackexchange_text_document',
    ]
    
    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    splits_string = '9999,1'
    ## 2. specify total samples needed
    ## 400B = 400 * 1000 * 1000 * 1000./ 2048 = 195312500
    ## 1000B = 1000 * 1000 * 1000 * 1000./ 2048 = 488281250
    train_max_num_samples = 195312500
    train_valid_test_num_samples = [train_max_num_samples, int(train_max_num_samples*0.00001)]
    seq_length = 2048
    seed = 2023
    skip_warmup = True

    train_dataset, valid_dataset, _ = _build_train_valid_test_weighted_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup,
        train_max_num_samples)
    print("Total train_dataset: ", len(train_dataset), flush=True)
    print("Total valid_dataset: ", len(valid_dataset), flush=True)
    
    def collate_fn(batch):
        def padding(indice, max_length, pad_idx=tokenizer.token_end_id):
            pad_indice = [
                item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
            ]
            return torch.tensor(pad_indice)
    
        input_ids = [data["input_ids"] for data in batch]
        max_length = max([len(t) for t in input_ids])
        input_ids = padding(input_ids, max_length)[:,:seq_length]
    
        data = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return data
    
    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=collate_fn,
        optimizer=None,
        rank_split=False)

else:
    data_prefix = [
        1.0,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_9_dedup_wudao_text_document',
        1.0,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_9_part_merged_text_document',
        1.0,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pile-cc_text_document',
        1.51,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-openwebtext2_text_document',
    
        0.6,
        '/data/indexed_dataset/batch1_tok100k_sep/code_dedup-md5-pile-github_text_document',
        0.53,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code_text_document',
        0.53,
        '/data/indexed_dataset/batch1_tok100k_sep/code_newcode1_text_document',
        0.53,
        '/data/indexed_dataset/batch1_tok100k_sep/code_newcode2_text_document',
        0.38,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code-cpp_text_document',
        0.38,
        '/data/indexed_dataset/batch1_tok100k_sep/code_code-java_text_document',
    
        1.06,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_baike_text_document',
        2.43,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-wikipedia_en_text_document',
    
        1.0,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_ebook_merge_maxlen_text_document',
        1.42,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-gutenberg_pg-19_text_document',
        1.42,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-bookcorpus2_text_document',
        1.42,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-books3_text_document',
        1.14,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-arxiv_text_document',
        1.14,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-pubmed_abstracts_text_document',
    
        1.13,
        '/data/indexed_dataset/batch1_tok100k_sep/cn_zhihu_text_document',
        2.08,
        '/data/indexed_dataset/batch1_tok100k_sep/en_dedup-md5-pile-stackexchange_text_document',
    ]
    
    data_impl = 'mmap'
    ## splits_string len should same as train_valid_test_num_samples len
    splits_string = '9999,1'
    ## rebuilding if no npy files for train_valid_test_num_samples config
    train_valid_test_num_samples = [195312500, 19531]
    seq_length = 2048
    seed = 2023
    skip_warmup = True
    ## 400 * 1000 * 1000 * 1000./ 2048 = 195312500
    train_max_num_samples = 195312500
    train_dataset, valid_dataset, _ = _build_train_valid_test_weighted_datasets(
        data_prefix, data_impl, splits_string,
        train_valid_test_num_samples,
        seq_length, seed, skip_warmup,
        train_max_num_samples)
    print("Total train_dataset: ", len(train_dataset), flush=True)
    print("Total valid_dataset: ", len(valid_dataset), flush=True)
    
    def collate_fn(batch):
        def padding(indice, max_length, pad_idx=tokenizer.token_end_id):
            pad_indice = [
                item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
            ]
            return torch.tensor(pad_indice)
    
        input_ids = [data["input_ids"] for data in batch]
        max_length = max([len(t) for t in input_ids])
        input_ids = padding(input_ids, max_length)[:,:seq_length]
    
        data = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return data
    
    trainer.do_train(
        train_dataset=train_dataset,
        valid_dataset=None,
        collate_fn=collate_fn,
        optimizer=None,
        rank_split=False)
