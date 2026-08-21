import json
import time
import random
import os
import csv

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import bmtrain as bmt

from model_center import get_args
from model_center.model import T5, T5Config
from model_center.tokenizer import T5Tokenizer
from model_center.dataset.t5dataset import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader

from bmcook.quant import *
from bmcook.utils import config as bmcook_config
from bmcook.moe import *
from bmcook.pruning import *

from loras import LoraModel


def set_UD(model, state: bool):
    for n, p in model.named_parameters():
        if 'lora_a1' in n or 'lora_a2' in n:
            p.requires_grad = state

def set_LoRA(model, state: bool):
    for n, p in model.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            p.requires_grad = state

def set_model_otherparam(model, state: bool):
    for n, p in model.named_parameters():
        if 'lora' not in n:
            p.requires_grad = state
    
def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=args.weight_decay)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    # get the model
    tokenizer = T5Tokenizer.from_pretrained(args.model_config)
    config = T5Config.from_pretrained(args.model_config)
    model = T5(config)
    base_model = T5(config)
    # bmt.init_parameters(model)
    # bmt.init_parameters(base_model)
    bmt.synchronize()

    save_name = args.save_name.split('-', maxsplit=1)[1]
    args.exp_kind = save_name
    bmt.print_rank("exp_kind: ", save_name)
    bmt.load(model, args.model_ckpt_path)
    bmt.load(base_model, args.model_ckpt_path)
    if args.comp_type == 'quant':
        bmt.load(model, args.quant_ckpt_path) 
        ckconfig = bmcook_config.ConfigParser(args.quant_config_path) 
        BMQuant.quantize(model,ckconfig) 
    elif args.comp_type == 'moe':
        bmt.load(model, args.model_ckpt_path) 
        ratio = 0.2
        with torch.no_grad():
            BMMoE.moefy(
                model, 
                512, 
                int(512*ratio), 
                24, 
                ['encoder.layers.{}.ffn.ffn.w_in.w.weight', 'decoder.layers.{}.ffn.ffn.w_in.w.weight'], 
                args.moe_ckpt_path)
        torch.cuda.empty_cache()
    elif args.comp_type == 'pr':
        pr = BMPrune()
        bmt.load(model, args.pr_ckpt_path) 
        ckconfig = bmcook_config.ConfigParser(args.pr_config_path)
        pr.compute_mask(model, ckconfig) 
    elif args.comp_type == 'spr':
        pr = BMPrune()
        bmt.load(model, args.spr_ckpt_path)
        ckconfig = bmcook_config.ConfigParser(args.spr_config_path)
        pr.compute_mask(model, ckconfig) 
    elif args.comp_type == 'mix':
        bmt.load(model, args.mix_ckpt_path)
        ckconfig = bmcook_config.ConfigParser(args.quant_config_path) 
        BMQuant.quantize(model,ckconfig) 
        ratio = 0.2
        with torch.no_grad():
            BMMoE.moefy(model,
                        512,
                        int(512*ratio), 
                        24,
                        ['encoder.layers.{}.ffn.ffn.w_in.w.weight', 'decoder.layers.{}.ffn.ffn.w_in.w.weight'],
                        args.mix_layer_ckpt_path)
        torch.cuda.empty_cache()
    bmt.synchronize()
    if args.pet == 'True': # add lora
        if args.recover == 'True': # add recover
            delta_model = LoraModel(
                backbone_model=model,
                modified_modules=['project_q', 'project_k'],
                lora_r=32,
                backend='bmt',
                lora_type='full',
            )
            delta2_model = LoraModel(
                backbone_model=model,
                modified_modules=['project_v', 'attention_out', 'w_in.w', 'w_out'],
                lora_r = 32, # TODO
                backend='bmt',
                lora_type='activate',
            )
            set_model_otherparam(model, False)
            if bmt.rank() == 0:
                delta2_model.log()
        else: # without recover
            delta_model = LoraModel(
                backbone_model=model,
                modified_modules=['project_q', 'project_k'],
                lora_r=32,
                backend='bmt',
                lora_type='normal',
            )
            set_model_otherparam(model, False)
            if bmt.rank() == 0:
                delta_model.log()
    else: # without lora
        if args.recover == 'True': # add recover
            delta_model = LoraModel(
                backbone_model=model,
                modified_modules=['project_q', 'project_k', 'project_v', 'attention_out', 'w_in.w', 'w_out'],
                lora_r = 32, # TODO
                backend='bmt',
                lora_type='activate',
            )
            set_model_otherparam(model, False)
            
            
    if args.pet_init_type == 'inherit':
        # initialise LoRA of compressed model by D
        from bmtrain.store import DistributedStateDictWrapper
        from collections import OrderedDict
        state_dict = torch.load(args.inherit_ckpt_path)
        cnt = 0
        for ith, (k, v) in enumerate(state_dict.items()):
            if 'lora' not in k:
                continue
            key, kk = k.rsplit('.', maxsplit=1)
            if cnt % 2 == 0:
                dd = OrderedDict()
                o = model
                for part in key.split('.'):
                    if part.isdigit(): o = o[int(part)]
                    else: o = getattr(o, part)
            dd[kk] = v
            if cnt % 2 == 1:
                dd = DistributedStateDictWrapper(dd if bmt.rank() == 0 else {})
                o.load_state_dict(dd, strict=False)
            cnt += 1
    
    if args.distill == "True":
        # add LoRA to M
        delta_model3 = LoraModel(
            backbone_model=base_model,
            modified_modules=['project_q', 'project_k'],
            lora_r=32,
            backend='bmt',
            lora_type='normal',
        )
        
        # plugin delta-tuned D to M
        from bmtrain.store import DistributedStateDictWrapper
        from collections import OrderedDict
        state_dict = torch.load(args.inherit_ckpt_path)
        cnt = 0
        for ith, (k, v) in enumerate(state_dict.items()):
            if 'lora' not in k:
                continue
            key, kk = k.rsplit('.', maxsplit=1)
            if cnt % 2 == 0:
                dd = OrderedDict()
                o = base_model
                for part in key.split('.'):
                    if part.isdigit(): o = o[int(part)]
                    else: o = getattr(o, part)
            dd[kk] = v
            if cnt % 2 == 1:
                dd = DistributedStateDictWrapper(dd if bmt.rank() == 0 else {})
                o.load_state_dict(dd, strict=False)
            cnt += 1

    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    
    if args.comp_type == 'pr' or args.comp_type == 'spr':
        pr.set_optim_for_pruning(optimizer)
        
    
    # get the memory usage
    # bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()

    return tokenizer, model, base_model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size):
    splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_encoder_length, args.max_decoder_length)
    verbalizer = torch.LongTensor(DATASET[dataset_name].get_verbalizer(tokenizer)).cuda()
    return dataset, verbalizer


def finetune(args, tokenizer, model, base_model, optimizer, lr_scheduler, dataset, verbalizer):
    
    state_dict = model.state_dict()
    if bmt.rank() == 0:
        torch.save(state_dict, os.path.join(args.save, args.save_name+f"test.pt"))

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    distil_func = torch.nn.MSELoss()

    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale, loss_scale_steps=100)
    optim_manager.add_optimizer(optimizer, lr_scheduler)


    bmt.print_rank(verbalizer)

    for epoch in range(20):
        dataloader = {
            "train": DistributedDataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True),
            "dev": DistributedDataLoader(dataset['dev'], batch_size=args.batch_size, shuffle=False),
        }

        model.train()
        for it, data in enumerate(dataloader['train']):
            enc_input = data["enc_input"]
            enc_length = data["enc_length"]
            dec_input = data["dec_input"]
            dec_length = data["dec_length"]
            targets = data["targets"]
            index = data["index"]

            output = model(enc_input, enc_length, dec_input, dec_length, output_logits=True)
            if args.distill == 'True':
                with torch.no_grad():
                    base_output = base_model(enc_input, enc_length, dec_input, dec_length, output_logits=True)
            logits = output.logits
            logits = logits.index_select(dim=-1, index=verbalizer)
            logits = logits[torch.where(index==1)]

            loss1 = loss_func(logits, targets)
            loss2 = 0
            loss3 = 0
            if args.distill == 'True':
                loss2 = distil_func(output.encoder_hidden_states[23].float(), base_output.encoder_hidden_states[23].float()).half()
                loss3 = distil_func(output.decoder_hidden_states[23].float(), base_output.decoder_hidden_states[23].float()).half()
            loss = loss1 + 0.005 * loss2 + 0.05 * loss3 
            global_loss = bmt.sum_loss(loss).item()

            optim_manager.zero_grad()

            optim_manager.backward(loss)
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type = 2)

            optim_manager.step()

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataloader["train"]),
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    grad_norm,
                )
            )
            
        state_dict = model.state_dict()
        if bmt.rank() == 0:
            torch.save(state_dict, os.path.join(args.save, args.save_name+f"{epoch}.pt"))

        model.eval()
        with torch.no_grad():
            for split in ['dev']:
                pd = []
                gt = []
                for it, data in enumerate(dataloader[split]):
                    enc_input = data["enc_input"]
                    enc_length = data["enc_length"]
                    dec_input = data["dec_input"]
                    dec_length = data["dec_length"]
                    targets = data["targets"]
                    index = data["index"]

                    logits = model(enc_input, enc_length, dec_input, dec_length, output_logits=True).logits
                    logits = logits.index_select(dim=-1, index=verbalizer)
                    logits = logits[torch.where(index==1)]
                    logits = logits.argmax(dim=-1)
                
                    pd.extend(logits.cpu().tolist())
                    gt.extend(targets.cpu().tolist())

                    bmt.print_rank(
                        "{} | epoch {:3d} | Iter: {:6d}/{:6d} |".format(
                            split,
                            epoch,
                            it,
                            len(dataloader[split]),
                        )
                    )
                pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
                gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()
                
                if args.dataset_name in ["BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"]:
                    acc = accuracy_score(gt, pd)
                    bmt.print_rank(f"{split} epoch {epoch}:", f"accuracy: {acc*100:.2f}")
                if args.dataset_name in ["CB"]:
                    f1 = f1_score(gt, pd, average="macro")
                    bmt.print_rank(f"{split} epoch {epoch}:", f"Average F1: {f1*100:.2f}")


def main():
    beg = time.time()
    args = initialize()
    tokenizer, model, base_model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset, verbalizer = prepare_dataset(
        args,
        tokenizer,
        f"/cdgm0705/hyx/down_data/superglue/",
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, base_model, optimizer, lr_scheduler, dataset, verbalizer)
    end = time.time()

if __name__ == "__main__":
    main()
