import os
import sys
sys.path.append(".")

import argparse
import csv
from re import template
from process_data import load_dataset
from dect_verbalizer import DecTVerbalizer
from dect_trainer import DecTRunner
from openprompt.prompts import ManualTemplate
from openprompt.pipeline_base import PromptForClassification
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.data_utils import FewShotSampler
from openprompt.plms import load_plm, LMTokenizerWrapper, T5LMTokenizerWrapper, T5TokenizerWrapper
from openprompt.data_utils.utils import InputExample
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, AutoModelForSeq2SeqLM, AutoConfig
import torch
parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='roberta', help="plm name")
parser.add_argument("--size", type=str, default='large', help="plm size")
parser.add_argument("--type", type=str, default='mlm', help="plm type")
parser.add_argument("--model_name_or_path", default='roberta-large', help="default load from Huggingface cache")
parser.add_argument("--shot", type=int, default=16, help="number of shots per class")
parser.add_argument("--seed", type=int, default=0, help="data sampling seed")
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--dataset", type=str, default='sst2')
parser.add_argument("--max_epochs", type=int, default=30, help="number of training epochs for DecT")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for train and test")
parser.add_argument("--proto_dim", type=int, default=128, help="hidden dimension for DecT prototypes")
parser.add_argument("--model_logits_weight", type=float, default=1, help="weight factor (\lambda) for model logits")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate for DecT")
args = parser.parse_args()

def load_model(name, size, path):
    if name == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(path)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        model = LlamaForCausalLM.from_pretrained(path)
        wrapper = LMTokenizerWrapper
        if size == "7b":
            hidden_size = 4096
        elif size == "13b":
            hidden_size = 5120
    elif name == "alpaca":
        tokenizer = LlamaTokenizer.from_pretrained(path)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(path)
        wrapper = LMTokenizerWrapper
        hidden_size = 4096
    elif name == "vicuna":
        tokenizer = LlamaTokenizer.from_pretrained(path)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(path)
        wrapper = LMTokenizerWrapper
        hidden_size = 5120
    else:
        model, tokenizer, model_config, wrapper = load_plm(args.model, args.model_name_or_path)
        hidden_size = model_config.hidden_size
    
    return model, tokenizer, hidden_size, wrapper


def build_dataloader(dataset, template, verbalizer, tokenizer, tokenizer_wrapper_class, batch_size):
    dataloader = PromptDataLoader(
        dataset = dataset, 
        template = template, 
        verbalizer = verbalizer, 
        tokenizer = tokenizer, 
        tokenizer_wrapper_class=tokenizer_wrapper_class, 
        decoder_max_length=128,
        batch_size = batch_size,
        shuffle = True,
    )

    return dataloader



def main():
    # set hyperparameter
    datasets = args.dataset.split(',')
    data_path = datasets[0].split('-')[0]
    if data_path == "mnli":
        args.model_logits_weight = 1
    elif data_path == "fewnerd":
        args.model_logits_weight = 1/16
    else:
        args.model_logits_weight = 1/args.shot

    # load dataset. The valid_dataset can be None
    
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(datasets[0])
    
    set_seed(123)
    # sample data
    sampler = FewShotSampler(
        num_examples_per_label = args.shot,
        also_sample_dev = True,
        num_examples_per_label_dev = args.shot)

    train_sampled_dataset, valid_sampled_dataset = sampler(
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        seed = args.seed
    )

    plm, tokenizer, hidden_size, plm_wrapper_class = load_model(args.model, args.size, args.model_name_or_path)
    # make dir to save hiddens and logits
    save_dir = f"vectors/{args.model}/{args.size}/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    # define template and verbalizer
    # define prompt
    template = ManualTemplate(
        tokenizer=tokenizer).from_file(f"scripts/{args.type}/{data_path}/manual_template.txt", choice=args.template_id)

    verbalizer = DecTVerbalizer(
        tokenizer=tokenizer, 
        classes=Processor.labels, 
        hidden_size=hidden_size, 
        lr=args.lr, 
        mid_dim=args.proto_dim, 
        epochs=args.max_epochs, 
        model_logits_weight=args.model_logits_weight,
        save_dir=save_dir).from_file(f"scripts/{args.type}/{data_path}/manual_verbalizer.json")
        
    # load prompt’s pipeline model
    prompt_model = PromptForClassification(plm, template, verbalizer)
            
    # process data and get data_loader
    train_dataloader = build_dataloader(train_sampled_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if train_dataset else None
    valid_dataloader = build_dataloader(valid_sampled_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size) if valid_dataset else None

    test_dataloader = build_dataloader(test_dataset, template, verbalizer, tokenizer, plm_wrapper_class, args.batch_size)

    calibrate_dataloader =  PromptDataLoader(
        dataset = [InputExample(guid=str(0), text_a="", text_b="", meta={"entity": "It"}, label=0)], 
        template = template, 
        tokenizer = tokenizer, 
        tokenizer_wrapper_class=plm_wrapper_class,
        decoder_max_length=128,
    )

    runner = DecTRunner(
        model = prompt_model,
        train_dataloader = train_dataloader,
        valid_dataloader = valid_dataloader,
        test_dataloader = test_dataloader,
        calibrate_dataloader = calibrate_dataloader,
        id2label = Processor.id2label,
        verbalizer = verbalizer
    )                                   

        
    res = runner.run()
    print('Dataset: {} | Shot: {} |'.format(args.dataset, args.shot))
    print(res)
    return res["dect acc"]
    


if __name__ == "__main__":
    res = []
    for seed in range(1):
        args.seed = seed
        res.append(main())
    print('Average: {}'.format(sum(res)/len(res)*100))
    
