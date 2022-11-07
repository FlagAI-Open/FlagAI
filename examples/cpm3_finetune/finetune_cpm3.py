import time
import random
import torch
import bmtrain as bmp
from bmtrain import nccl
from bmtrain.global_var import config
import numpy as np
import os


from flagai.model.cpm3_train_model import CPM3
from flagai.data.tokenizer.cpm_3 import CPM3Tokenizer
from flagai.model.cpm3_train_model import CPM3Config



from flagai. data.dataset.cpm3_data import DistributedMMapIndexedDataset, MMapIndexedDataset, CPM3_Dataset_Merge
from arguments import get_args
import errno
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
    config = CPM3Config.from_json_file(args.model_config)
    print ("vocab size:%d"%(config.vocab_size))

    model = CPM3.from_pretrain(model_name = 'cpm3-train', download_path='/sharefs/baai-mrnd/xw/')

    return model

def get_optimizer(args, model):
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = bmp.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
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

def batch_iter(args, dataset, start_step = 0, batch_size=2, concat=False, shuffle=True):
    st = 0
    ctx = []
    tgt = []
    context = []
    position = []
    segment = []
    span = []

    exist_total = 0

    idx = [i for i in range(len(dataset))]
    if shuffle:
        random.shuffle(idx) # 每个epoch都shuffle一遍数据

    while st < len(dataset): 
        ctx_data, tgt_data, _len, context_data, position_data, segment_data, _ = dataset[idx[st]]
    # while True:
        # ctx_data, tgt_data, _len, context_data, position_data, segment_data = dataset[st]
        st += 1
        if ctx_data is None:
            continue
        assert _len <= args.max_length

        ctx_data = ctx_data.astype("int64")
        tgt_data = tgt_data.astype("int64")

        # finetune不拼数据
        if not concat:
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
            ctx.append(_ctx)
            tgt.append(_tgt)
            context.append(_context)
            position.append(_position)
            segment.append(_segment)
            span.append([_len])
        else:
            for index in range(len(ctx)):
                if span[index][-1] + _len < args.max_length:
                    ctx[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(ctx_data)[:_len].long()
                    tgt[index][span[index][-1]:span[index][-1] + _len]= torch.from_numpy(tgt_data)[:_len].long()
                    context[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(context_data)[:_len].bool()
                    position[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(position_data)[:_len].float()
                    segment[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(segment_data)[:_len].long()
                    span[index].append(span[index][-1] + _len) # span里是多个list，每个list里面是诸如5，15，26，37的递增的总长度
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
                ctx.append(_ctx)
                tgt.append(_tgt)
                context.append(_context)
                position.append(_position)
                segment.append(_segment)
                span.append([_len])

        # 拼数据时，这里应该使用大于
        if len(ctx) >= batch_size:
            if exist_total >= start_step:
                _span = torch.zeros((batch_size, args.max_length + 1), dtype=torch.long)
                for bindex in range(batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1 # 每个拼接的句子结尾的后一位是1
                yield {
                    "ctx": torch.stack(ctx[:batch_size]),
                    "tgt": torch.stack(tgt[:batch_size]),
                    "context": torch.stack(context[:batch_size]),
                    "segment": torch.stack(segment[:batch_size]),
                    "position": torch.stack(position[:batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:,:-1], # cumsum：前缀和
                    "len_ctx": torch.LongTensor([it[-1] for it in span[:batch_size]]),
                }
            exist_total += 1
            ctx = ctx[batch_size:]
            tgt = tgt[batch_size:]
            context = context[batch_size:]
            segment = segment[batch_size:]
            position = position[batch_size:]
            span = span[batch_size:]

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


def finetune(args, tokenizer, model, optimizer, lr_scheduler, train_dataset, eval_dataset):
    average_time = 0
    average_time_shift = 0.9
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step
    iteration = start_step

    if not args.is_nlu:
        best_eval_metric = 1e9
    else:
        best_eval_metric = -1e9
    best_eval_step = 0
    model.to(torch.device('cuda', args.local_rank))
    print_inspect(model, "*")

    if bmp.rank() == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(int(args.epochs)):
        for data in batch_iter(args, train_dataset, start_step, args.batch_size, concat=False, shuffle=True):
            model.train()
            
            iteration += 1

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

            logits, _ = model(input_idx, input_length, input_context, input_position, input_segment, input_span)

            loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
            global_loss = bmp.sum_loss(loss).item()

            loss = optimizer.loss_scale(loss)
            loss.backward()

            grad_norm = clip_grad_norm(optimizer.param_groups, 1.0, scale = optimizer.scale / config['world_size'], norm_type = 2)

            bmp.optim_step(optimizer, lr_scheduler)

            iteration_time = time.time() - st
            average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time

            bmp.print_rank(
                "| Train Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} | token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}".format(
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
            
            if iteration % args.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    total_metric = 0
                    eval_batch_size = min(len(eval_dataset), args.eval_batch_size)
                    cnt = 0
                    for idx, data in enumerate(batch_iter(args, eval_dataset, 0, eval_batch_size, concat=False, shuffle=False)):
                        assert len(data["ctx"]) == eval_batch_size

                        input_idx = data["ctx"].int().cuda()
                        input_length = data["len_ctx"].int().cuda()
                        input_context = data["context"].bool().cuda()
                        input_position = data["position"].float().cuda()
                        input_segment = data["segment"].int().cuda()
                        input_span = data["span"].int().cuda()
                        targets = data["tgt"].long().cuda()

                        logits, _ = model(input_idx, input_length, input_context, input_position, input_segment, input_span)
                        
                        # debug
                        # Note: this is teaching force manner!
                        if idx == 0:
                            logits_max = torch.max(logits, -1)[1]
                            batch_size = logits.size(0)
                            for i in range(batch_size):
                                target_ids = targets[i]
                                pred_ids = logits_max[i]
                                target_mask = target_ids.ne(-100)
                                valid_target_ids = torch.masked_select(target_ids, target_mask).tolist()
                                valid_pred_ids = torch.masked_select(pred_ids, target_mask).tolist()
                                target = tokenizer.decode(valid_target_ids)
                                pred = tokenizer.decode(valid_pred_ids)
                                bmp.print_rank("target: ", target)
                                bmp.print_rank("pred: ", pred)
                        
                        if not args.is_nlu:
                            metric = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
                        else:
                            # acc
                            logits_max = torch.max(logits, -1)[1]
                            masked_logits_max = torch.masked_fill(logits_max, targets.eq(-100), 0)
                            masked_targets = torch.masked_fill(targets, targets.eq(-100), 0)
                            metric = (masked_logits_max == masked_targets).all(dim=-1).sum().float() / logits.size(0)
                        
                        total_metric += bmp.sum_loss(metric).item()
                        cnt += 1
                    
                    total_metric /= cnt
                    bmp.print_rank(
                    "| Eval Iter: {:6d} | Metric: {:.4f}".format(
                        iteration,
                        total_metric
                        )
                    )

                    if (not args.is_nlu and total_metric < best_eval_metric) or (args.is_nlu and total_metric > best_eval_metric):
                        bmp.save(model, os.path.join(args.save, args.save_name + "-best_eval.pt"))
                        bmp.print_rank("Iteration {} is the best checkpoint now!".format(iteration))
                        best_eval_metric = total_metric
                        best_eval_step = iteration
                    elif args.early_stop_patience !=-1 and (iteration - best_eval_step) // args.eval_step >= args.early_stop_patience:
                        bmp.print_rank("early stop at iteration {}!".format(iteration))
                        exit(0)
                    if bmp.rank() == 0:
                        writer.add_scalar("Metric/eval", total_metric, iteration)
            
            if iteration % args.inspect_iters == 0:
                print_inspect(model, "*")
            
            if bmp.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
            
            if args.save != None and iteration % args.save_iters == 0:
                bmp.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % iteration)))

def main():
    args = initialize()

    # assert args.dataset is not None, "Must specify a dataset when tuning model!"
    assert args.load is not None, "Must load a checkpoint when tuning model!"

    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    # dataset_main_path = "/sharefs/baai-mrnd/xw/final_cpm3".format(args.dataset)
    dataset_main_path = "/sharefs/baai-mrnd/xw/final_cpm3/"
    if not os.path.exists(dataset_main_path):
        raise FileNotFoundError( 
            errno.ENOENT, os.strerror(errno.ENOENT), dataset_main_path)

    # 训练集
    # train_dataset = CPM3_Dataset_Merge(
    #     DistributedMMapIndexedDataset(os.path.join(dataset_main_path, "train/"), "cpm3_lm_document_context", bmp.rank(), bmp.world_size()),
    #     DistributedMMapIndexedDataset(os.path.join(dataset_main_path, "train/"), "cpm3_lm_document_target", bmp.rank(), bmp.world_size()),
    #     args.max_length
    # )

    train_dataset = CPM3_Dataset_Merge(
        DistributedMMapIndexedDataset(os.path.join(dataset_main_path), "cpm3_lm_document_context", bmp.rank(), bmp.world_size()),
        DistributedMMapIndexedDataset(os.path.join(dataset_main_path), "cpm3_lm_document_target", bmp.rank(), bmp.world_size()),
        args.max_length
    )

    # 验证集
    # eval_dataset = CPM3_Dataset_Merge(
    #     DistributedMMapIndexedDataset(os.path.join(dataset_main_path, "eval/"), "cpm3_lm_document_context", bmp.rank(), bmp.world_size()),
    #     DistributedMMapIndexedDataset(os.path.join(dataset_main_path, "eval/"), "cpm3_lm_document_target", bmp.rank(), bmp.world_size()),
    #     args.max_length
    # )

    eval_dataset =CPM3_Dataset_Merge(
        DistributedMMapIndexedDataset(os.path.join(dataset_main_path), "cpm3_lm_document_context", bmp.rank(), bmp.world_size()),
        DistributedMMapIndexedDataset(os.path.join(dataset_main_path), "cpm3_lm_document_target", bmp.rank(), bmp.world_size()),
        args.max_length
    )

    finetune(args, tokenizer, model, optimizer, lr_scheduler, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
