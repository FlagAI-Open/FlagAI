from bmtrain.optim import optim_manager
from utils import *

from typing import Optional
import torch
import math
import torch.nn.functional as F
import bmtrain as bmt
from bmtrain.global_var import config
import os

class Attention(torch.nn.Module):
    def __init__(self, 
            dim_model : int, dim_head : int,
            num_heads : int, bias : bool = True,
            dtype = None
        ) -> None:
        super().__init__()

        self.project_q = torch.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_k = torch.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_v = torch.nn.Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)

        self.project_out = torch.nn.Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_model = dim_model
    
    def forward(self, 
            hidden_q : torch.Tensor,        # (batch_size, seq_q, dim_model)
            hidden_kv : torch.Tensor,       # (batch_size, seq_kv, dim_model)
            mask : torch.BoolTensor,        # (batch_size, seq_q, seq_kv)
            position_bias : Optional[torch.Tensor] = None,   # (batch, num_heads, seq_q, seq_kv)
        ) -> torch.Tensor:
        batch_size, seq_q, dim_model = hidden_q.size()
        seq_kv = hidden_kv.size(1)

        h_q : torch.Tensor = self.project_q(hidden_q)
        h_k : torch.Tensor = self.project_k(hidden_kv)
        h_v : torch.Tensor = self.project_v(hidden_kv)

        h_q = h_q.view(batch_size, seq_q, self.num_heads, self.dim_head)
        h_k = h_k.view(batch_size, seq_kv, self.num_heads, self.dim_head)
        h_v = h_v.view(batch_size, seq_kv, self.num_heads, self.dim_head)

        h_q = h_q.permute(0, 2, 1, 3).contiguous()
        h_k = h_k.permute(0, 2, 1, 3).contiguous()
        h_v = h_v.permute(0, 2, 1, 3).contiguous()

        h_q = h_q.view(batch_size * self.num_heads, seq_q, self.dim_head)
        h_k = h_k.view(batch_size * self.num_heads, seq_kv, self.dim_head)
        h_v = h_v.view(batch_size * self.num_heads, seq_kv, self.dim_head)

        score = torch.bmm(
            h_q, h_k.transpose(1, 2)
        )
        score = score / math.sqrt(self.dim_head)

        score = score.view(batch_size, self.num_heads, seq_q, seq_kv)

        if position_bias is not None:
            score = score + position_bias.view(batch_size, self.num_heads, seq_q, seq_kv)
        
        score = torch.where(
            mask.view(batch_size, 1, seq_q, seq_kv),
            score,
            torch.scalar_tensor(float('-inf'), device=score.device, dtype=score.dtype)
        )

        score = torch.where(
            mask.view(batch_size, 1, seq_q, seq_kv),
            self.softmax(score),
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )

        score = score.view(batch_size * self.num_heads, seq_q, seq_kv)

        h_out = torch.bmm(
            score, h_v
        )
        h_out = h_out.view(batch_size, self.num_heads, seq_q, self.dim_head)
        h_out = h_out.permute(0, 2, 1, 3).contiguous()
        h_out = h_out.view(batch_size, seq_q, self.num_heads * self.dim_head)

        attn_out = self.project_out(h_out)
        return attn_out
        
class Feedforward(torch.nn.Module):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = True, dtype = None) -> None:
        super().__init__()

        self.w_in = torch.nn.Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
        self.w_out = torch.nn.Linear(dim_ff, dim_model, bias = bias, dtype=dtype)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return self.w_out(self.relu(self.w_in(input)))


class TransformerEncoder(torch.nn.Module):
    def __init__(self,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.ln_attn = torch.nn.LayerNorm(dim_model, dtype=dtype)
        self.attn = Attention(dim_model, dim_head, num_heads, bias=bias, dtype=dtype)

        self.ln_ff = torch.nn.LayerNorm(dim_model, dtype=dtype)
        self.ff = Feedforward(dim_model, dim_ff, bias=bias, dtype=dtype)
    
    def forward(self,
            hidden : torch.Tensor,      # (batch, seq_len, dim_model)
            mask : torch.BoolTensor,    # (batch, seq_len, dim_model)
            position_bias : Optional[torch.Tensor] = None,   # (batch, num_head, seq_len, seq_len)
        ):
        x = self.ln_attn(hidden)
        x = self.attn(x, x, mask, position_bias)
        hidden = hidden + x

        x = self.ln_ff(hidden)
        x = self.ff(x)
        hidden = hidden + x

        return hidden
    

class GPT(torch.nn.Module):
    def __init__(self,
            num_layers : int, vocab_size : int,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            max_distance : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.dtype = dtype
        self.max_distance = max_distance

        self.word_emb = torch.nn.Embedding(vocab_size, dim_model, dtype=dtype)
        self.pos_emb = torch.nn.Embedding(max_distance, dim_model, dtype=dtype)
        self.dim_model = dim_model
        
        self.transformers = torch.nn.ModuleList([
            TransformerEncoder(
                dim_model, dim_head, num_heads, dim_ff, bias, dtype
            )
            for _ in range(num_layers)
        ])

        self.layernorm = torch.nn.LayerNorm(dim_model, dtype=dtype)

    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
        ) -> torch.Tensor:

        mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])

        input_emb = self.pos_emb(pos) + self.word_emb(input)

        out = input_emb
        if isinstance(self.transformers, torch.nn.ModuleList):
            for layer in self.transformers:
                out = layer(out, mask_2d, None)
        else:
            out = self.transformers(out, mask_2d, None)
        out = self.layernorm(out)

        logits = F.linear(out, self.word_emb.weight) / math.sqrt(self.dim_model)

        return logits

def sub_train_torch(model, loss_func_cls, optimizer_cls):
    loss_func = loss_func_cls(ignore_index=-100)
    optimizer = optimizer_cls(model.parameters(), weight_decay=1e-2)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    optim_manager = bmt.optim.OptimManager(loss_scale=2**20 if model.dtype == torch.half else None)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    # use the break if i == bmt.rank() to generate different data on different rank
    torch.manual_seed(2333)
    batch_size = 2
    seq_len = 512

    sents = []
    enc_lengths = []
    enc_inputs = []
    targetss = []
    masks = []
    for i in range(bmt.world_size()):
        sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
        enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
        enc_input = sent[:, :-1].long().cuda()
        targets = sent[:, 1:].long().cuda()
        mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
        targets = torch.where(
            mask,
            targets,
            torch.full_like(targets, -100, dtype=torch.long)
        )

        sents.append(sent)
        enc_lengths.append(enc_length)
        enc_inputs.append(enc_input)
        targetss.append(targets)
        masks.append(mask)

    sent = torch.cat(sents, dim=0)
    enc_length = torch.cat(enc_lengths, dim=0)
    enc_input = torch.cat(enc_inputs, dim=0)
    targets = torch.cat(targetss, dim=0)
    mask = torch.cat(masks, dim=0)

    pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)

    logs = []
    for iter in range(100):
        optim_manager.zero_grad()

        logits = model(enc_input, pos, pos < enc_length[:, None])

        batch, seq_len, vocab_out_size = logits.size()

        loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
    
        global_loss = loss.item()

        loss = optim_manager.loss_scale * loss
        loss.backward()

        grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=10.0)

        optim_manager.step()

        bmt.print_rank("| Iter: {:6d} | loss: {:.4f} {:.4f} | lr: {:.4e} scale: {:10.4f} | grad_norm: {:.4f} |".format(
            iter,
            global_loss,
            loss,
            lr_scheduler.current_lr,
            optim_manager.loss_scale,
            grad_norm,
        ))
        logs.append(global_loss)

    summary = bmt.inspect.inspect_model(model, "*")
    return logs, summary

def sub_train(model, loss_func_cls, optimizer_cls):
    loss_func = loss_func_cls(ignore_index=-100)
    optimizer = optimizer_cls(model.parameters(), weight_decay=1e-2)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    optim_manager = bmt.optim.OptimManager(loss_scale=2**20 if model.dtype == torch.half else None)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    # use the break if i == bmt.rank() to generate different data on different rank
    torch.manual_seed(2333)
    batch_size = 2
    seq_len = 512

    for i in range(bmt.world_size()):
        sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
        enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
        enc_input = sent[:, :-1].long().cuda()
        targets = sent[:, 1:].long().cuda()
        mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
        targets = torch.where(
            mask,
            targets,
            torch.full_like(targets, -100, dtype=torch.long)
        )

        if i == bmt.rank():
            break

    pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)

    logs = []
    for iter in range(100):
        optim_manager.zero_grad()

        logits = model(enc_input, pos, pos < enc_length[:, None])

        batch, seq_len, vocab_out_size = logits.size()

        loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
    
        global_loss = bmt.sum_loss(loss).item()

        optim_manager.backward(loss)

        grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=10.0)

        optim_manager.step()

        bmt.print_rank("| Iter: {:6d} | loss: {:.4f} {:.4f} | lr: {:.4e} scale: {:10.4f} | grad_norm: {:.4f} |".format(
            iter,
            global_loss,
            loss,
            lr_scheduler.current_lr,
            optim_manager.loss_scale,
            grad_norm,
        ))
        logs.append(global_loss)

    summary = bmt.inspect.inspect_model(model, "*")
    return logs, summary
    
def train(model, loss_func, optimizer):
    key = f"{model[0]}*{loss_func[0]}*{optimizer[0]}"
    model = model[1]()
    if key.startswith("torch"):
        ret = sub_train_torch(model, loss_func[1], optimizer[1])
    else:
        ret = sub_train(model, loss_func[1], optimizer[1])
    del model
    bmt.print_rank(f"finished {key}")
    return key, ret

def test_main(test_fp16=True, test_fp32=True):
    ckpt_path = "test_ckpt.pt"

    kwargs = {
        "num_layers": 8,
        "vocab_size": 10240, 
        "dim_model": 2560,
        "dim_head": 80,
        "num_heads": 32,
        "dim_ff": 8192,
        "max_distance": 1024,
        "bias": True,
        "dtype": None,
    }

    def make_ref_ckpt():
        model = GPT(**kwargs)
        if bmt.rank() == 0:
            torch.save(model.state_dict(), ckpt_path)
        bmt.synchronize()
        del model

    ret = {}
    def torch_model():
        model = GPT(**kwargs)
        model.load_state_dict(torch.load(ckpt_path))
        model = model.cuda()
        return model

    def wrap_model():
        model = GPT(**kwargs)
        wrap_model = bmt.BMTrainModelWrapper(model)
        bmt.load(wrap_model, ckpt_path)
        return model

    def list_model():
        model = GPT(**kwargs)
        list_model = bmt.BMTrainModelWrapper(model)
        list_model.transformers = bmt.TransformerBlockList([m for m in list_model.transformers])
        bmt.load(list_model, ckpt_path)
        return model

    def pipe_model():
        model = GPT(**kwargs)
        pipe_model = bmt.BMTrainModelWrapper(model)
        for m in pipe_model.transformers:
            assert isinstance(m, bmt.CheckpointBlock)
        pipe_model.transformers = bmt.PipelineTransformerBlockList([m for m in pipe_model.transformers])
        bmt.load(pipe_model, ckpt_path)
        return model

    models = {
        "torch": torch_model,
        "wrapper": wrap_model,
        "blocklist": list_model,
        "pipelist": pipe_model,
    }
    loss_funcs = {
        "bmt_entropy": bmt.loss.FusedCrossEntropy,
        "torch_entropy": torch.nn.CrossEntropyLoss,
    }
    optimizers = {
        "bmt_adam": bmt.optim.AdamOptimizer,
        "bmt_adam_offload": bmt.optim.AdamOffloadOptimizer,
        "torch_adam": torch.optim.Adam,
    }

    ret = {}
    def add_to_check_list(m, l, o):
        key, value = train((m, models[m]), (l, loss_funcs[l]), (o, optimizers[o]))
        ret[key] = value

    if test_fp16:
        kwargs["dtype"] = torch.half
        make_ref_ckpt()
        add_to_check_list("torch", "bmt_entropy", "bmt_adam")
        add_to_check_list("wrapper", "bmt_entropy", "bmt_adam")
        add_to_check_list("blocklist", "bmt_entropy", "bmt_adam")
        add_to_check_list("pipelist", "bmt_entropy", "bmt_adam")
        add_to_check_list("blocklist", "torch_entropy", "bmt_adam")
        add_to_check_list("blocklist", "bmt_entropy", "bmt_adam_offload")
        if bmt.rank() == 0:
            os.remove(ckpt_path)
        check(ret)

    if test_fp32:
        kwargs["dtype"] = torch.float
        make_ref_ckpt()
        add_to_check_list("torch", "torch_entropy", "bmt_adam")
        add_to_check_list("wrapper", "torch_entropy", "bmt_adam")
        add_to_check_list("blocklist", "torch_entropy", "bmt_adam")
        add_to_check_list("pipelist", "torch_entropy", "bmt_adam")
        add_to_check_list("blocklist", "torch_entropy", "bmt_adam_offload")
        add_to_check_list("blocklist", "torch_entropy", "torch_adam")
        if bmt.rank() == 0:
            os.remove(ckpt_path)
        check(ret)

def check(ret):
    if bmt.rank() == 0:
        for k1, v1 in ret.items():
            for k2, v2 in ret.items():
                print(f"checking {k1} vs. {k2}")
                check_param(v1[1], v2[1])
    bmt.synchronize()
    ret.clear()

def check_param(info1, info2):
    for l1, l2 in zip(info1, info2):
        for key in ['std', 'mean', 'max', 'min']:
            v1 = l1[key]
            v2 = l2[key]
            assert_lt(abs(v1-v2), 1e-2)

if __name__ == '__main__':
    bmt.init_distributed(pipe_size=2)

    test_main(test_fp16=True, test_fp32=True)