import sys
import os
import torch
from torch.utils.data import Dataset
import argparse
import time
import wandb
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
#from flagai.data.tokenizer.llama.tokenizer import Tokenizer
parser = argparse.ArgumentParser()
parser.add_argument("--student_model_path", default="galactica-6.7b-en", type=str, help="path of the pretrained student model")
parser.add_argument("--save_path", default="current_training/checkpoints", type=str, help="path to save checkpoints")
#parser.add_argument("--train_file", default="/share/project/lijijie/dataset/pile_merged/pile_1000_text_document", type=str, help="path to train file")
parser.add_argument("--train_file", default="/share/project/ldwang/data/indexed_dataset/batch1_tok100k/cn_zhihu_text_document", type=str, help="path to train file")
parser.add_argument("--host_file", default="./hostfile.llama_ds_mpu", type=str, help="path to eval file")
# parser.add_argument("--data_scripts", default="data_script/json.py", type=str, help="path to save checkpoints")
# parser.add_argument("--data_cache", default="data_cache/", type=str, help="path to save checkpoints")
parser.add_argument("--log_dir", default="./current_training/llama_node16_mp4_bs4_alldatav1/", type=str, help="path to save checkpoints")
parser.add_argument("--epochs", default=100, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=1, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=1.5e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="The episilon for Adam.")
parser.add_argument("--clip_grad_norm", default=1.0, type=float, help="The truncated grad threshold")
parser.add_argument("--log_every_n_step", default=1, type=int, help="The steps to output training information.")
parser.add_argument("--save_every_n_steps", default=5000, type=int, help="The epochs to save the trained models.")
parser.add_argument("--max_seq_length", default=2048, type=int, help="The maximum length of input sentences")
parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="The gradient_accumulation_steps.")
parser.add_argument("--not_call_launch", default=False, action="store_true")
parser.add_argument("--world_size", default=16*8, type=int)
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()
args.hostfile = './hostfile'
args.per_gpu_batch_size = 1
args.log_every_n_step = 1

def main():

    trainer = Trainer(
        env_type="deepspeed+mpu",
        epochs=args.epochs,
        batch_size=args.per_gpu_batch_size,
        eval_interval=1e12,
        log_interval=args.log_every_n_step,
        save_interval=args.save_every_n_steps,
        pytorch_device='cuda:0',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        experiment_name='llama_bs_mpu',
        load_dir=None,
        lr=args.learning_rate,
        tensorboard_dir = 'current_training/tboard',
        master_ip=os.getenv("MASTER_IP"),
        master_port=int(os.getenv("MASTER_PORT")),
        num_gpus=1,
        num_nodes=1,
        fp16=True,
        save_optim=True,
        save_rng = True,
        save_dir=args.save_path,
        checkpoint_activations=True,
        hostfile=args.host_file,
        training_script=__file__,
        deepspeed_config='deepspeed.json',
        model_parallel_size=1,
        )
    
    # tokenizer = Tokenizer.from_pretrained('./state_dict/llama-30b-en/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/share/project/liuguang/flagai-internal/examples/gpt2_title_generation/state_dict/llama-7b-en')
    model_dir = "state_dict"
    model_dir = "/share/project/ldwang/checkpoints/"
    auto_loader = AutoLoader(
    "lm",
    model_name="llama-7b-en",
    model_dir=model_dir,
    only_download_config=True,
    use_cache=False,
    checkpoint_activations=True,
    use_fp16 = True
    )
    model = auto_loader.get_model()
    tokenizer = auto_loader.get_tokenizer()
    print('*'*20, "tokenizer", tokenizer)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = cur_dir + '/data/train.src'
    tgt_dir = cur_dir + '/data/train.tgt'
    maxlen = 256
    
    def read_file():
        src = []
        tgt = []
    
        with open(src_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                src.append(line.strip('\n').lower())
    
        with open(tgt_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tgt.append(line.strip('\n').lower())
    
        return src, tgt
    
    class GPT2Seq2seqDataset(Dataset):
    
        def __init__(self, sents_src, sents_tgt, tokenizer, maxlen=21):
            super(GPT2Seq2seqDataset, self).__init__()
            self.sents_src = sents_src
            self.sents_tgt = sents_tgt
            self.tokenizer = tokenizer
            self.maxlen = maxlen
    
        def __getitem__(self, i):
            src = self.sents_src[i][:512]
            tgt = self.sents_tgt[i]
            in_text = f"{src}。对以上文字提取重点:{tgt}"
            
            data = self.tokenizer.encode(in_text, True, True)
    
            output = {
                "input_ids": data,
            }
            return output
    
        def __len__(self):
    
            return len(self.sents_src)

    def distill_collate_fn(batch):
        def padding(indice, max_length, pad_idx=1):
            pad_indice = [
                #item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
                item + [pad_idx] * max(0, max_length - len(item)) for item in indice
            ]
            return torch.tensor(pad_indice)

        input_ids = [data["input_ids"] for data in batch]
        max_length = max([len(t) for t in input_ids])
        input_ids = padding(input_ids, max_length,0)[:,:args.max_seq_length]

        data = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return data

    sents_src, sents_tgt = read_file()
    data_len = len(sents_tgt)
    train_size = int(data_len * 0.8)
    
    train_src = sents_src[:train_size]
    train_tgt = sents_tgt[:train_size]
    
    train_dataset = GPT2Seq2seqDataset(train_src,
                                       train_tgt,
                                       tokenizer=tokenizer,
                                       maxlen=maxlen)

    trainer.train(
        model=model,
        train_dataset=train_dataset,
        # valid_dataset=eval_dataset,
        collate_fn=distill_collate_fn,
    )

if __name__ == '__main__':
    main()
