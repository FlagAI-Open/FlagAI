import argparse
import bmtrain as bmt
from model_center.model import Bert, BertConfig
import time
import torch
import torch.nn as nn

model_config = {
    "1.8b": {
        "dim_model": 1024,
        "num_heads": 32,
        "dim_ff": 16384,
        "num_layers": 48,
    },
    "6b": {
        "dim_model": 1024,
        "num_heads": 128,
        "dim_ff": 65536,
        "num_layers": 48,
    },
    "13b": {
        "dim_model": 2048,
        "num_heads": 128,
        "dim_ff": 65536,
        "num_layers": 48,
    },
    "100b": {
        "dim_model": 12288,
        "num_heads": 96,
        "dim_ff": 12288 * 4,
        "num_layers": 56,
    },
    "175b": {
        "dim_model": 12288,
        "num_heads": 96,
        "dim_ff": 12288 * 4,
        "num_layers": 96,
    },
}

class TestModel(bmt.DistributedModule):
    def __init__(self, model_size):
        super().__init__()
        self.config = BertConfig(
            vocab_size = 30522,
            dim_model = model_config[model_size]["dim_model"],
            num_heads = model_config[model_size]["num_heads"],
            dim_head = model_config[model_size]["dim_model"] // model_config[model_size]["num_heads"],
            dim_ff = model_config[model_size]["dim_ff"],
            num_layers = model_config[model_size]["num_layers"],
        )
        self.bert = Bert(self.config)
        bmt.init_parameters(self.bert)

    def forward(self, input_ids, *args, **kwargs):
        out = self.bert(input_ids=input_ids, output_pooler_output = False)
        return out.last_hidden_state[:, 0, :]
    
    def get_tbl_optim(self):
        if self.bert.encoder.layers._auto_optimization is not None:
            auto_optim = self.bert.encoder.layers._auto_optimization
            optim = auto_optim.optimization
            runtime = auto_optim._estimated_runtime
            memory = auto_optim._estimated_memory
            save_path = auto_optim.save_path
            return optim, runtime, memory, save_path
        return None, None, None, None

class Timer:
    def __init__(self):
        self.total = 0
        self.start_time = None
    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
    def pause(self):
        assert self.start_time is not None
        torch.cuda.synchronize()
        self.total += time.time() - self.start_time
        self.start_time = None
    def end(self):
        if self.start_time is not None:
            self.pause()
        ret = self.total
        self.total = 0
        return ret

def train(
        batch_size,
        steps = 128,
        print_period = 32,
        model_size = "base",
        log_prefix = "",
    ):

    learning_rate = 5e-5
    max_seq_len = 512

    model = TestModel(model_size).cuda()
    model.train()
    criterion = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), lr = learning_rate)
    optim_manager = bmt.optim.OptimManager()
    optim_manager.add_optimizer(optimizer)

    total_time = 0.
    total_time_f = 0.
    total_time_b = 0.
    total_time_o = 0.
    total_time_steps = 0
    last_checkpoint_step = 0

    for i in range(steps):
        x = torch.randint(model.config.vocab_size, (batch_size, max_seq_len), dtype = torch.long, device = "cuda")
        label = torch.randint(model.config.dim_model, (batch_size, ), dtype=torch.long, device = "cuda")
        timer = Timer()

        timer.start()
        y = model(x)
        t_f = timer.end()

        loss = criterion(y.contiguous(), label)
        optim_manager.zero_grad()

        timer.start()
        optim_manager.backward(loss)
        t_b = timer.end()

        timer.start()
        optim_manager.step()
        t_o = timer.end()
        t = t_f + t_b + t_o

        total_time += t
        total_time_f += t_f
        total_time_b += t_b
        total_time_o += t_o
        total_time_steps += 1
        _time_f = round(total_time_f / total_time_steps, 3)
        _time_b = round(total_time_b / total_time_steps, 3)
        _time_o = round(total_time_o / total_time_steps, 3)
        _time = "{}s = {} + {} + {}".format(round(total_time / total_time_steps, 3), _time_f, _time_b, _time_o)
        _memory = torch.cuda.max_memory_allocated()
        if _memory >= 2 ** 30:
            _memory = "{}GiB".format(round(_memory / (2 ** 30), 1))
        else:
            _memory = "{}MiB".format(round(_memory / (2 ** 20), 1))
        _loss = round(float(loss), 4)

        obuf = log_prefix + f"step = {last_checkpoint_step}~{i} | time = {_time} | memory = {_memory} | loss = {_loss}"
        bmt.print_rank(obuf, end='\r')
        if (i + 1) % print_period == 0:
            total_time = 0.
            total_time_f = 0.
            total_time_b = 0.
            total_time_o = 0.
            total_time_steps = 0
            last_checkpoint_step = i
            bmt.print_rank("")
            if bmt.config["rank"] == 0:
                with open("result.log.txt", "a") as f:
                    f.write(obuf + '\n')
    if bmt.config["rank"] == 0:
        optim, runtime, memory, save_path = model.get_tbl_optim()
        obuf = f"Estimated runtime: {runtime} sec. Estimated memory: {memory} bytes." + "\n"
        obuf += f"Optimization file saved path: {save_path}"
        bmt.print_rank(obuf)
        with open("result.log.txt", "a") as f:
            f.write(obuf + '\n')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--alg-name", type=str, required=True)
    parser.add_argument("--gpu-name", type=str, default="a100")
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    model_size = args.model_size
    alg_name = args.alg_name
    gpu_name = args.gpu_name
    batch_size_global = args.batch_size
    assert model_size in model_config.keys()
    assert batch_size_global % 8 == 0

    if gpu_name == "2080ti":
        memory_limit = 8000 * (2 ** 20)
        nvlink_available = False
    elif gpu_name == "a100":
        memory_limit = 28 * (2 ** 30)
        nvlink_available = True
    else:
        raise NotImplementedError

    if alg_name == "wo_h3t_solver":
        bmt.init_distributed(
            seed=0,
            nvlink_available = nvlink_available,
            zero_level = 3,
            offload_parameter = True,
            checkpointing = True,
            offload_hidden_state = True,
            tbl_auto_optimization = False,
        )
    else:
        bmt.init_distributed(
            seed=0,
            nvlink_available = nvlink_available,
            tbl_auto_optimization = alg_name,
            tbl_memory_limit = memory_limit,
        )

    train(
        batch_size = batch_size_global // bmt.config["world_size"],
        model_size = model_size,
        log_prefix = f"world_size = {bmt.config['world_size']} | model_size = {model_size} | batch_size = {batch_size_global} | alg_name = {alg_name} | ",
    )
