import os
import sys
import torch
from torch.utils.data import Dataset
import gc
gc.collect()
torch.cuda.empty_cache()
import flash_attn
from flagai.data.tokenizer import Tokenizer
from tqdm import tqdm

from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_datasets
from examples.gpt3_pretrain.build_index_mappings import _build_train_valid_test_weighted_datasets
from examples.aquila.gpt import GPTLMHeadModel, combine_state_dicts_tp
from flash_attn.models.llama import remap_state_dict_meta_llama, llama_config_to_gpt2_config
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint

assert len(sys.argv) >= 3
model_checkpoints = sys.argv[1]
model_iters = sys.argv[2]
model_list = [os.path.join(model_checkpoints, model_iters)]

## Just share same configs of tokenizer & model
checkpoints = '/data2/ldwang/checkpoints'
shared_model_name = 'Aquila-7b'
cache_dir = os.path.join(checkpoints, shared_model_name)
tokenizer = Tokenizer.from_pretrained(shared_model_name, cache_dir=cache_dir)

config = llama_config_to_gpt2_config(config_from_checkpoint(checkpoints, shared_model_name))
config.vocab_size=100008
config.use_cache = False
config.attn_pdrop = 0.0
config.resid_pdrop = 0.0
config.layer_norm_epsilon = 1e-5

config.fused_bias_fc = False
config.fused_mlp = False  # We don't have fused GatedMLP yet
config.fused_dropout_add_ln = False
config.residual_in_fp32 = False
config.use_flash_attn = True

if True:
    max_seq_len = 2048
    IGNORE_INDEX = -100

    import jsonlines
    import numpy as np
    def read_file(jsonl_file):
        conversations = []
        with jsonlines.open(jsonl_file) as reader:
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
                    role_labels = [IGNORE_INDEX] * len(content)
                labels += role_labels

            example.append(EOS_TOKEN)
            labels.append(EOS_TOKEN)

            ## delete bos & eos
            #example = example[1:-1]
            #labels = labels[1:-1]

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
            labels = padding(labels, max_length, pad_idx=IGNORE_INDEX)[:,:max_length]
    
            data = {
                "input_ids": input_ids,
                "labels": labels
            }
            return data
    
    jsonl_data_val = '/data/ldwang/sft_datasets/convo_v2/sft_data_val_v0.84.jsonl'
    valid_dataset = None
    if jsonl_data_val is not None:
        conversations_val = read_file(jsonl_data_val)
        valid_dataset = ConversationDatasetV2(conversations_val,
                                              tokenizer=tokenizer,
                                              maxlen=max_seq_len)
    import torch
    dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        collate_fn=ConversationDatasetV2.collate_fn,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False)

model = GPTLMHeadModel(config, device='cuda:1', dtype=torch.float16)
for ck in model_list:
    checkpoint_path = os.path.join(f'{ck}', "pytorch_model.bin")
    ckpt_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt_model, strict=True)
    gc.collect()
    torch.cuda.empty_cache()
    losses = []
    accuracy = []
    model.eval()
    for d in tqdm(dataloader):
        batch = {x: d[x].to(next(model.parameters()).device)for x in d}
        output = model.forward(**batch)
        losses += output.loss.view(-1).detach().cpu().numpy().tolist()
        accuracy += output.acc.view(-1).detach().cpu().numpy().tolist()
    print(f"{ck} {sum(losses)/len(losses)}")

    with open(f"{shared_model_name}_sft_loss.log", 'a') as wf:
        wf.write(f"{shared_model_name} {ck} {sum(losses)/len(losses)} {sum(accuracy)/len(losses)}\n")

