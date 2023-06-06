import sys
sys.path.append('/share/project/liuguang/flagai-internal')
# sys.path.append('/share/project/liuguang/Megatron-LM')
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    OPTForCausalLM
    )
import os
import torch
import argparse
import time
from build_index_mappings import _build_train_valid_test_weighted_datasets
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
from flagai.trainer import Trainer
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
#from flagai.data.tokenizer.llama.tokenizer import Tokenizer
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--student_model_path", default="galactica-6.7b-en", type=str, help="path of the pretrained student model")
parser.add_argument("--save_path", default="current_training/checkpoints", type=str, help="path to save checkpoints")
#parser.add_argument("--train_file", default="/share/project/lijijie/dataset/pile_merged/pile_1000_text_document", type=str, help="path to train file")
parser.add_argument("--train_file", default="/share/project/ldwang/data/indexed_dataset/batch1_tok100k/cn_zhihu_text_document", type=str, help="path to train file")
parser.add_argument("--host_file", default="./current_training/hostfile_llama", type=str, help="path to eval file")
# parser.add_argument("--data_scripts", default="data_script/json.py", type=str, help="path to save checkpoints")
# parser.add_argument("--data_cache", default="data_cache/", type=str, help="path to save checkpoints")
parser.add_argument("--log_dir", default="./current_training/llama_node16_mp4_bs4_alldatav1/", type=str, help="path to save checkpoints")
parser.add_argument("--epochs", default=1, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=4, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=1.5e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="The episilon for Adam.")
parser.add_argument("--clip_grad_norm", default=1.0, type=float, help="The truncated grad threshold")
parser.add_argument("--log_every_n_step", default=16, type=int, help="The steps to output training information.")
parser.add_argument("--save_every_n_steps", default=5000, type=int, help="The epochs to save the trained models.")
parser.add_argument("--max_seq_length", default=2048, type=int, help="The maximum length of input sentences")
parser.add_argument("--gradient_accumulation_steps", default=16, type=int, help="The gradient_accumulation_steps.")
parser.add_argument("--world_size", default=16*8, type=int)

parser.add_argument("--not_call_launch", default=False, action="store_true")
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()

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
        experiment_name='llama_node16_zero2_mp4_bs4_alldatav1',
        load_dir=None,
        lr=args.learning_rate,
        tensorboard_dir = 'current_training/tboard',
        master_ip=os.getenv("MASTER_IP"),
        master_port=int(os.getenv("MASTER_PORT")),
        num_gpus=8,
        num_nodes=1,
        fp16=True,
        save_optim=True,
        save_rng = True,
        save_dir=args.save_path,
        checkpoint_activations=True,
        hostfile=args.host_file,
        training_script=__file__,
        deepspeed_config='deepspeed.json',
        model_parallel_size=8,
        extra_args=parser,
        )
    
    # tokenizer = Tokenizer.from_pretrained('./state_dict/llama-30b-en/tokenizer.model')
    # tokenizer = AutoTokenizer.from_pretrained('/share/project/liuguang/flagai-internal/examples/gpt2_title_generation/state_dict/llama-7b-en')
    model_dir = "/data/ldwang/state_dict/"
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

    # TODO
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
    print(len(train_dataset))
    print(len(valid_dataset)) 

    def distill_collate_fn(batch):
        def padding(indice, max_length, pad_idx=1):
            pad_indice = [
                item.tolist() + [pad_idx] * max(0, max_length - len(item.tolist())) for item in indice
            ]
            return torch.tensor(pad_indice)

        input_ids = [data["input_ids"] for data in batch]
        max_length = max([len(t) for t in input_ids])
        input_ids = padding(input_ids, max_length,0)[:,:seq_length]

        data = {
            "input_ids": input_ids,
            "labels": input_ids
        }
        return data
    print("training data size:",len(train_dataset))
    trainer.train(
        model=model,
        train_dataset=train_dataset,
        # valid_dataset=eval_dataset,
        collate_fn=distill_collate_fn,
    )

if __name__ == '__main__':
    main()
