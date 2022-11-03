import time
import random
import torch
try:
    import bmtrain as bmp
except:
    pass
from bmtrain import nccl
from bmtrain.global_var import config
import numpy as np
import os
from flagai.model.cpm3_train_model import CPM3Config, CPM3
from flagai.data.tokenizer.cpm_3 import CPM3Tokenizer
from flagai.data.dataset.cpm3_data import DistributedMMapIndexedDataset, CPM3_Dataset_Merge
from arguments import get_args
import distutils.version
from torch.utils.tensorboard import SummaryWriter

task_ids = {
        'lm':0,
        'compress':1,
        'expand':2,
        'rewrite':3,
        'rewrite_s':4,
        'compress_para':5,
        'expand_para':6,
}

def get_tokenizer(args):
    tokenizer = CPM3Tokenizer(args.vocab_file)
    return tokenizer

def get_model(args):
    config = CPM3Config._dict_from_json_file(args.model_config)
    print ("vocab size:%d"%(config['vocab_size']))

    if args.load != None:
        model = CPM3.from_pretrain(model_name = 'cpm3-train', download_path='/sharefs/baai-mrnd/xw/')
    else:
        model = CPM3(config)
        bmp.init_parameters(model)
    return model

def get_optimizer(args, model):
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = bmp.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
                                        
    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    bmp.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmp.synchronize()
    # get the memory usage
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def batch_iter(args, dataset, start_step = 0):
    st = 0
    ctx = []
    tgt = []
    context = []
    position = []
    segment = []
    span = []
    task_info = []

    exist_total = 0
    while True:
        ctx_data, tgt_data, _len, context_data, position_data, segment_data, task_data = dataset[st]
        st += 1
        if ctx_data is None:
            continue
        assert _len <= args.max_length

        ctx_data = ctx_data.astype("int64")
        tgt_data = tgt_data.astype("int64")

        for index in range(len(ctx)):
            if span[index][-1] + _len < args.max_length:
                ctx[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(ctx_data)[:_len].long()
                tgt[index][span[index][-1]:span[index][-1] + _len]= torch.from_numpy(tgt_data)[:_len].long()
                context[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(context_data)[:_len].bool()
                position[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(position_data)[:_len].float()
                segment[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(segment_data)[:_len].long()
                task_info[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(task_data)[:_len].long()
                span[index].append(span[index][-1] + _len)
                break
        else:
            _ctx = torch.zeros((args.max_length,), dtype=torch.long)
            _ctx[:_len] = torch.from_numpy(ctx_data)[:_len].long()
            _tgt = torch.full((args.max_length,), -100, dtype=torch.long)
            _tgt[:_len] = torch.from_numpy(tgt_data)[:_len].long()
            _context = torch.full((args.max_length,), False, dtype=torch.bool)
            _context[:_len] = torch.from_numpy(context_data)[:_len].bool()
            _position = torch.full((args.max_length,), False, dtype=torch.float)
            _position[:_len] = torch.from_numpy(position_data)[:_len].float()
            _segment = torch.full((args.max_length,), False, dtype=torch.long)
            _segment[:_len] = torch.from_numpy(segment_data)[:_len].long()
            _task_info = torch.full((args.max_length,), -1, dtype=torch.long)
            _task_info[:_len] = torch.from_numpy(task_data)[:_len].long()
            ctx.append(_ctx)
            tgt.append(_tgt)
            context.append(_context)
            position.append(_position)
            segment.append(_segment)
            task_info.append(_task_info)
            span.append([_len])

        if len(ctx) > args.batch_size:
            if exist_total >= start_step:
                _span = torch.zeros((args.batch_size, args.max_length + 1), dtype=torch.long)
                for bindex in range(args.batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1
                
                yield {
                    "ctx": torch.stack(ctx[:args.batch_size]),
                    "tgt": torch.stack(tgt[:args.batch_size]),
                    "context": torch.stack(context[:args.batch_size]),
                    "segment": torch.stack(segment[:args.batch_size]),
                    "position": torch.stack(position[:args.batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:,:-1],
                    "len_ctx": torch.LongTensor([it[-1] for it in span[:args.batch_size]]),
                    "task": torch.stack(task_info[:args.batch_size]),
                }
            exist_total += 1
            ctx = ctx[args.batch_size:]
            tgt = tgt[args.batch_size:]
            context = context[args.batch_size:]
            segment = segment[args.batch_size:]
            position = position[args.batch_size:]
            span = span[args.batch_size:]
            task_info = task_info[args.batch_size:]

def print_inspect(model, name):
    bmp.print_rank(
        bmp.inspect.format_summary(
            bmp.inspect.inspect_model(model, name)
        )
    )

def clip_grad_norm(param_groups, max_norm, scale, norm_type=2, eps=1e-6):

    parameters = [p for group in param_groups for p in group['params'] if p.grad is not None]

    if norm_type == 'inf':
        total_norm_cuda = max(p.grad.data.abs().max() for p in parameters).detach()
        nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "max", config["comm"])
        total_norm = total_norm_cuda
    else:
        norm_type = float(norm_type)
        total_norm_cuda = torch.cuda.FloatTensor([0])
        for p in parameters:
            param_norm = p.grad.data.float().norm(norm_type)
            total_norm_cuda += param_norm ** norm_type
        nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "sum", config["comm"])
        total_norm = total_norm_cuda[0] ** (1. / norm_type)

    clip_coef = float(max_norm * scale) / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm / scale


def pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    average_time = 0
    average_time_shift = 0.9
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)
    model.to(torch.device('cuda', args.local_rank))
    start_step = args.start_step

    if bmp.rank() == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    for iteration, data in enumerate(batch_iter(args, dataset, start_step)):
        iteration = iteration + start_step + 1

        st = time.time()
        optimizer.zero_grad()

        assert len(data["ctx"]) == args.batch_size

        input_idx = data["ctx"].int().cuda()
        input_length = data["len_ctx"].int().cuda()
        input_context = data["context"].bool().cuda()
        input_position = data["position"].float().cuda()
        input_segment = data["segment"].int().cuda()
        input_span = data["span"].int().cuda()
        targets = data["tgt"].long().cuda()
        task_info = data["task"].long().cuda()
        logits, _ = model(input_idx, input_length, input_context, input_position, input_segment, input_span)

        with torch.no_grad():
            task_num = len(task_ids)
            logits_tmp = logits.view(-1, logits.size(-1)).expand(task_num, -1, -1)
            targets_tmp = targets.expand(task_num, -1, -1)
            task_info = task_info.expand(task_num, -1, -1)

            task = task_info.new([x for x in range(task_num)])[:, None, None]
            targets_tmp = torch.where(task_info == task, targets_tmp, -100)

            task_loss_list = []
            for i in range(task_num):
                task_loss = loss_func(logits_tmp[i, :], targets_tmp[i, :].view(-1))
                global_task_loss = bmp.gather_result(task_loss.unsqueeze(0)).nanmean().item()
                task_loss_list.append(global_task_loss)

        loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
        global_loss = bmp.sum_loss(loss).item()

        loss = optimizer.loss_scale(loss)
        loss.backward()
        
        grad_norm = clip_grad_norm(optimizer.param_groups, 1.0, scale = optimizer.scale / config['world_size'], norm_type = 2)
    
        bmp.optim_step(optimizer, lr_scheduler)

        iteration_time = time.time() - st
        average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time

        bmp.print_rank(
            "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} | token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}".format(
                iteration,
                global_loss,
                lr_scheduler.current_lr,
                int(optimizer.scale),
                average_time / (1 - pow(average_time_shift, iteration + 1)),
                input_length.float().mean()/args.max_length,
                (targets>=0).sum(-1).float().mean()/args.max_length,
                grad_norm
            )
        )
        bmp.print_rank(
            "| " + " | ".join(["{} loss: {:.4f}".format(task_name, task_loss_list[idx]) for task_name, idx in task_ids.items()])
        )
        if iteration % args.inspect_iters == 0:
            print_inspect(model, "*")

        if bmp.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, iteration)
            for i in task_ids.keys():
                writer.add_scalar("Loss/train/{}".format(i), task_loss_list[task_ids[i]], iteration)
        
        if args.save != None and iteration % args.save_iters == 0:
            bmp.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % iteration)))

    bmp.save(model, os.path.join(args.save, args.save_name+".pt"))

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = CPM3_Dataset_Merge(
        DistributedMMapIndexedDataset("/sharefs/baai-mrnd/xw/final_cpm3/", "cpm3_lm_document_context", bmp.rank(), bmp.world_size()),
        DistributedMMapIndexedDataset("/sharefs/baai-mrnd/xw/final_cpm3/", "cpm3_lm_document_target", bmp.rank(), bmp.world_size()),
        args.max_length
    )
    pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
