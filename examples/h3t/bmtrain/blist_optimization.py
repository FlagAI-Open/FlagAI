from .block_optimization import BlockOptimization, max_block_optim, encode_block_optim, validate_boptim
from .global_var import config
from .utils import print_rank
from collections.abc import Iterable
import warnings
import torch
import time
import json
from .distributed import all_gather
from itertools import product
import copy
import os

# Scheduling Algorithms

def _all_valid_block_optimization():
    optims = []
    code2index = {}
    for zero_level in [2, 3]:
        for offload_parameter in [0, 1]:
            for checkpointing in [0, 1]:
                for offload_hidden_state in [0, 1]:
                    for economical_forward in [0, 1]:
                        for economical_backward in [0, 1]:
                            for segment_synchronization in [0, 1]:
                                try:
                                    bo = BlockOptimization(
                                        zero_level = zero_level,
                                        offload_parameter = offload_parameter,
                                        checkpointing = checkpointing,
                                        offload_hidden_state = offload_hidden_state,
                                        economical_forward = economical_forward,
                                        economical_backward = economical_backward,
                                        segment_synchronization = segment_synchronization,
                                    )
                                    bo = validate_boptim(bo)
                                except:
                                    continue
                                optims.append(bo)
                                code2index[encode_block_optim(bo)] = len(optims) - 1
    return optims, code2index
_block_optim_list, _optim_code2index = _all_valid_block_optimization()
_n_block_optim = len(_block_optim_list)
def _tensor2optim(tensor):
    if tensor is None:
        return None
    optim = []
    for i in range(tensor.shape[0]):
        if tensor[i] < 0 or tensor[i] >= _n_block_optim:
            return None
        optim.append(_block_optim_list[tensor[i]])
    return optim
def _optim2tensor(optim):
    ret = []
    for o in optim:
        ret.append(_optim_code2index[encode_block_optim(o)])
    ret = torch.LongTensor(ret)
    return ret

def optim_tensor_all_reduce(optim_tensor, runtime):
    if config["world_size"] == 1:
        return optim_tensor
    n = optim_tensor.shape[0]
    optim_tensor = torch.LongTensor(optim_tensor).cuda()
    optim_tensor = all_gather(optim_tensor).cpu()
    runtime = torch.Tensor([runtime]).cuda()
    runtime = all_gather(runtime).cpu()
    valid_optim_tensor = None
    for i in range(optim_tensor.shape[0]):
        o = _tensor2optim(optim_tensor[i])
        if o is not None:
            if valid_optim_tensor is None:
                valid_optim_tensor = optim_tensor[i]
            else:
                valid_optim_tensor = 2
    if valid_optim_tensor is None:
        return None
    elif torch.is_tensor(valid_optim_tensor):
        return valid_optim_tensor
    if config["rank"] == 0:
        found = False
        best_optim_tensor, best_runtime = None, None
        for i in range(runtime.shape[0]):
            if not found or float(runtime[i]) < best_runtime:
                best_optim_tensor = optim_tensor[i]
                best_runtime = float(runtime[i])
                found = True
    else:
        best_optim_tensor = torch.LongTensor([-1] * n)
        best_runtime = 1e9
    return optim_tensor_all_reduce(best_optim_tensor, best_runtime)

def random_scheduling(n, profile_runtime, profile_memory, memory_limit, rand_times = 10**3):
    found = False
    best_optim_tensor, best_runtime = torch.LongTensor([-1] * n), 1e9
    for _ in range(rand_times):
        optim_tensor = torch.randint(0, _n_block_optim, (config["world_size"], n))
        optim_tensor = optim_tensor[config["rank"]]
        optim = _tensor2optim(optim_tensor)
        assert optim is not None
        model_sim = ModelSimulator(
            layer_runtime = profile_runtime,
            layer_memory = profile_memory,
            layer_optim = optim,
        ).simulate_train()
        if model_sim.peak_memory > memory_limit:
            continue
        if not found or model_sim.max_runtime < best_runtime:
            best_optim_tensor = optim_tensor
            best_runtime = model_sim.max_runtime
    return best_optim_tensor, best_runtime

def greedy_scheduling(n, profile_runtime, profile_memory, memory_limit):
    if config["rank"] == 0:
        best_optim = [max_block_optim()] * n
        best_runtime = ModelSimulator(
            layer_runtime = profile_runtime,
            layer_memory = profile_memory,
            layer_optim = best_optim,
        ).simulate_train().max_runtime
        best_n_layer_optim = 0
        for n_layer_optim in range(1, n + 1):
            for layer_optim in _block_optim_list:
                optim = []
                gap = n // n_layer_optim
                for i in range(n_layer_optim):
                    optim.append(layer_optim)
                    optim.extend([max_block_optim()] * (gap - 1))
                optim.extend([max_block_optim()] * (n - len(optim)))
                model_sim = ModelSimulator(
                    layer_runtime = profile_runtime,
                    layer_memory = profile_memory,
                    layer_optim = optim,
                ).simulate_train()
                if model_sim.peak_memory > memory_limit:
                    continue
                if model_sim.max_runtime < best_runtime:
                    best_optim = optim
                    best_runtime = model_sim.max_runtime
                    best_n_layer_optim = n_layer_optim
        best_optim_tensor = _optim2tensor(best_optim)
        print_rank("[ greedy_scheduling ] best_layer_optim =", best_optim[0])
        print_rank("[ greedy_scheduling ] n_layer_optim =", best_n_layer_optim)
        print_rank("[ greedy_scheduling ] best_optim_tensor =", best_optim_tensor)
    else:
        best_optim_tensor = torch.LongTensor([-1] * n)
        best_runtime = 1e9
    return best_optim_tensor, best_runtime


def dp_scheduling(n, profile_runtime, profile_memory, memory_limit, n_memory = 128):
    if config["rank"] == 0:
        eps = 1e-6
        inf = 1e9
        n_state = 64
        def discretize_memory(mem):
            if mem > memory_limit:
                return -1
            if mem < 0:
                mem = 0
            return n_memory - int((memory_limit - mem) * n_memory // memory_limit)
        def serialize_memory(m):
            if m > n_memory or m < 0:
                return memory_limit + 1
            return int(memory_limit * m // n_memory) # ceil
        def encode_state(prev_optim, curr_optim):
            code = 0
            if prev_optim is not None:
                code = code << 1 ^ (prev_optim["zero_level"] - 2)
                code = code << 1 ^ prev_optim["offload_parameter"]
                code = code << 1 ^ prev_optim["offload_hidden_state"]
            code = code << 1 ^ (curr_optim["zero_level"] - 2)
            code = code << 1 ^ curr_optim["offload_parameter"]
            code = code << 1 ^ curr_optim["offload_hidden_state"]
            return code
        def init_dp_array():
            dp = []
            for s in range(n_state):
                dp.append([])
                for m_delta in range(n_memory + 1):
                    dp[-1].append([])
                    for m_peak in range(n_memory + 1):
                        dp[-1][-1].append(None)
            return dp
        def dp_pruning(dp):
            retained_solutions = []
            for s in range(n_state):
                for m_delta in range(n_memory + 1):
                    for m_peak in range(n_memory + 1):
                        if m_delta > 0 and dp[s][m_delta - 1][m_peak] is not None:
                            prev_sol_0 = dp[s][m_delta - 1][m_peak]
                        else:
                            prev_sol_0 = None
                        if m_peak > 0 and dp[s][m_delta][m_peak - 1] is not None:
                            prev_sol_1 = dp[s][m_delta][m_peak - 1]
                        else:
                            prev_sol_1 = None

                        if prev_sol_0 is None:
                            prev_sol = prev_sol_1
                        elif prev_sol_1 is None or prev_sol_0["runtime"] < prev_sol_1["runtime"]:
                            prev_sol = prev_sol_0
                        else:
                            prev_sol = prev_sol_1

                        curr_sol = dp[s][m_delta][m_peak]
                        if curr_sol is not None:
                            if prev_sol is None or curr_sol["runtime"] + eps < prev_sol["runtime"]:
                                curr_sol["m_delta"] = m_delta
                                curr_sol["m_peak"] = m_peak
                                retained_solutions.append(curr_sol)
                            else:
                                dp[s][m_delta][m_peak] = prev_sol
            return retained_solutions

        # Preparation
        suffix_max_resident = [0]
        for i in reversed(range(1, n)):
            suffix_max_resident.append(suffix_max_resident[-1] + profile_memory[i]["partitioned_parameter"])
        suffix_max_resident = list(reversed(suffix_max_resident))
        model_sim = ModelSimulator(
            layer_runtime = profile_runtime,
            layer_memory = profile_memory,
            layer_optim = [],
            auto_resident = False,
        )
        dp = init_dp_array()
        for zero, op, oh in product([2, 3], [0, 1], [0, 1]):
            optim = BlockOptimization(
                zero_level = zero,
                offload_parameter = op,
                offload_hidden_state = oh,
                segment_synchronization = True,
            )
            model_sim.layer_optim = [optim]

            model_sim.simulate_forward(i = 0, head = True)
            model_sim.stream_synchronize()
            runtime = model_sim.max_runtime
            mem_resident = model_sim.get_resident_memory(i = 0)
            mem_delta = model_sim.delta_memory + mem_resident
            mem_peak = model_sim.peak_memory + mem_resident

            mem_delta_backward = mem_delta - profile_memory[i]["partitioned_gradient"]
            model_sim.simulate_backward(i = 0, tail = True)
            model_sim.stream_synchronize()
            runtime += model_sim.max_runtime
            mem_peak = max(mem_peak, model_sim.peak_memory + mem_delta_backward)

            if mem_peak <= mem_delta or mem_peak + suffix_max_resident[0] <= memory_limit:
                mem_peak = 0
            if mem_peak > memory_limit or mem_delta > memory_limit:
                continue
            m_delta = discretize_memory(mem_delta)
            m_peak = discretize_memory(mem_peak)

            sol = {
                "runtime": runtime,
                "optim_seq": [optim]
            }
            state = encode_state(None, optim)
            dp[state][m_delta][m_peak] = sol

        # Dynamic Programming
        for i in range(1, n):
            dp_prev = dp_pruning(dp)
            dp = init_dp_array()
            print_rank(f"[ dp_scheduling ] Calculating for layer {i - 1}. {len(dp_prev)} solutions from previous layers.")
            for sol_prev in dp_prev:
                mem_delta_prev = serialize_memory(sol_prev["m_delta"])
                mem_peak_prev = serialize_memory(sol_prev["m_peak"])
                runtime_prev = sol_prev["runtime"]
                optim_seq_prev = sol_prev["optim_seq"]
                for ckpt, ef, eb in product([0, 1], [0, 1], [0, 1]):
                    if optim_seq_prev[-1]["offload_hidden_state"] and not ckpt:
                        continue
                    optim_seq = copy.copy(optim_seq_prev)
                    optim_seq[-1] = copy.deepcopy(optim_seq[-1])
                    optim_seq[-1]["checkpointing"] = ckpt
                    optim_seq[-1]["economical_forward"] = ef
                    optim_seq[-1]["economical_backward"] = eb
                    validate_boptim(optim_seq[-1])
                    optim_seq.append(None)
                    for zero, op, oh in product([2, 3], [0, 1], [0, 1]):
                        model_sim.layer_optim = copy.copy(optim_seq)
                        model_sim.layer_optim[i] = BlockOptimization(
                            zero_level = zero,
                            offload_parameter = op,
                            offload_hidden_state = oh,
                            segment_synchronization = True,
                        )
                        mem_resident = model_sim.get_resident_memory(i = i)
                        mem_peak = mem_peak_prev + mem_resident

                        model_sim.simulate_forward(i = i - 1)
                        model_sim.stream_synchronize()
                        runtime = runtime_prev + model_sim.max_runtime
                        mem_delta = mem_delta_prev + mem_resident + model_sim.delta_memory
                        mem_peak = max(mem_peak, model_sim.peak_memory + mem_delta_prev + mem_resident)

                        mem_delta_backward = mem_delta + 2 * profile_memory[i]["input_tensor"] + profile_memory[i]["gathered_gradient"]
                        if model_sim.layer_optim[i - 1]["checkpointing"] and not model_sim.layer_optim[i - 1]["offload_hidden_state"]:
                            mem_delta_backward -= profile_memory[i - 1]["input_tensor"]
                        model_sim.simulate_backward(i = i - 1)
                        model_sim.stream_synchronize()
                        runtime += model_sim.max_runtime
                        mem_peak = max(mem_peak, model_sim.peak_memory + mem_delta_backward)

                        if mem_peak <= mem_delta or mem_peak + suffix_max_resident[i] <= memory_limit:
                            mem_peak = 0
                        if mem_peak > memory_limit or mem_delta > memory_limit:
                            continue

                        model_sim.simulate_optimizer(i = i - 1)
                        model_sim.stream_synchronize()
                        runtime += model_sim.max_runtime
                        m_delta = discretize_memory(mem_delta)
                        m_peak = discretize_memory(mem_peak)

                        sol = {
                            "runtime": runtime,
                            "optim_seq": model_sim.layer_optim
                        }
                        state = encode_state(model_sim.layer_optim[i - 1], model_sim.layer_optim[i])
                        if dp[state][m_delta][m_peak] is None:
                            dp[state][m_delta][m_peak] = sol
                        elif dp[state][m_delta][m_peak]["runtime"] > runtime:
                            dp[state][m_delta][m_peak] = sol
        
        # Ending
        dp_prev = dp_pruning(dp)
        result = None
        for sol_prev in dp_prev:
            mem_delta_prev = serialize_memory(sol_prev["m_delta"])
            # print("m =", sol_prev["m_delta"], " mem = ", mem_delta_prev, " m =", discretize_memory(mem_delta_prev))
            runtime_prev = sol_prev["runtime"]
            optim_seq_prev = sol_prev["optim_seq"]
            for ckpt, ef, eb in product([0, 1], [0, 1], [0, 1]):
                if optim_seq_prev[-1]["offload_hidden_state"] and not ckpt:
                    continue
                optim_seq = copy.copy(optim_seq_prev)
                optim_seq[-1] = copy.deepcopy(optim_seq[-1])
                optim_seq[-1]["checkpointing"] = ckpt
                optim_seq[-1]["economical_forward"] = ef
                optim_seq[-1]["economical_backward"] = eb
                validate_boptim(optim_seq[-1])
                model_sim.layer_optim = optim_seq
                
                model_sim.simulate_forward(i = n - 1)
                model_sim.simulate_forward(i = n - 1, clear = False, tail = True)
                model_sim.stream_synchronize()
                model_sim.simulate_backward(i = n - 1, clear = False, head = True)
                model_sim.simulate_backward(i = n - 1, clear = False)
                model_sim.stream_synchronize()
                runtime = runtime_prev + model_sim.max_runtime
                mem_delta = mem_delta_prev + model_sim.delta_memory
                mem_peak = model_sim.peak_memory + mem_delta_prev

                if mem_peak > memory_limit or mem_delta > memory_limit:
                    continue

                model_sim.simulate_optimizer(i = n - 1)
                runtime += model_sim.max_runtime

                sol = {
                    "runtime": runtime,
                    "optim_seq": model_sim.layer_optim,
                }

                if result is None or runtime < result["runtime"]:
                    result = sol
        if result is not None:
            best_optim = result["optim_seq"]
            best_runtime = result["runtime"]
        else:
            best_optim = [max_block_optim()] * n
            best_runtime = 1e9
        best_optim_tensor = _optim2tensor(best_optim)
        print_rank("[ dp_scheduling ] best_optim_tensor =", best_optim_tensor)
        print_rank("[ dp_scheduling ] estimated runtime =", best_runtime)
    else:
        best_optim_tensor = torch.LongTensor([-1] * n)
        best_runtime = 1e9
    return best_optim_tensor, best_runtime


algorithm_map = {
    "default": dp_scheduling,
    "random": random_scheduling,
    "greedy": greedy_scheduling,
    "dp": dp_scheduling,
}

def get_scheduling_algorithm(alg_name):
    if not isinstance(alg_name, str):
        alg_name = "default"
    alg_name = alg_name.strip().lower()

    if alg_name not in algorithm_map:
        raise ValueError(f"Unknown scheduling algorithm {alg_name}.")
    return algorithm_map[alg_name], alg_name

class TBLAutoOptimization:
    def __init__(self, tbl, conf = None):
        self.tbl = tbl
        if conf is None:
            conf = config["tbl_auto_optimization"]

        self.memory_limit = conf["memory_limit"]
        self.scheduling_algorithm, self.alg_name = get_scheduling_algorithm(conf["algorithm"])
        self.kwargs = conf["kwargs"]

        self.convergent = {}
        self.profiling = {}
        self.training = self.tbl.training
        for i in self.tbl._modules.keys():
            self.tbl._modules[i].optimization = max_block_optim()
            self.convergent[i] = False
            self.profiling[i] = self.training
            if self.training:
                self.tbl._modules[i].profile.switch_on_()
            else:
                self.tbl._modules[i].profile.switch_off_()
        self.optimize_scheduled = False
        self.train_steps = -1
        self.max_profile_step = 50

    def refresh_convergent(self, refresh_profile = True):
        for i in self.tbl._modules.keys():
            self.convergent[i] = self.tbl._modules[i].profile.convergence()
        if not refresh_profile:
            return
        for i in self.tbl._modules.keys():
            if not self.convergent[i] and self.train_steps <= self.max_profile_step:
                self.profiling[i] = True
                self.tbl._modules[i].profile.switch_on_()
            else:
                self.profiling[i] = False
                self.tbl._modules[i].profile.switch_off_()

    def train(self):
        if self.training:
            return
        self.training = True
    
    def eval(self):
        if self.training:
            return
        self.training = False
        for i in self.tbl._modules.keys():
            self.profiling[i] = False
            self.tbl._modules[i].profile.switch_off_()

    def is_profiling(self, i = None):
        if not self.training or self.optimize_scheduled:
            return False
        if i is not None:
            i = str(i)
            return self.profiling[i]
        for i in self.tbl._modules.keys():
            if self.profiling[i]:
                return True
        return False

    def before_step(self):
        if not self.tbl.training or not torch.is_grad_enabled():
            self.eval()
            return
        self.train_steps += 1
        self.train()
        if self.optimize_scheduled:
            return
        self.refresh_convergent(refresh_profile = True)
        if self.train_steps > self.max_profile_step:
            if config["local_rank"] == 0:
                warnings.warn(f"TBLAutoOptimization has already collected model profiles for over {self.max_profile_step} steps, but some of them are still non-convergent. TBLAutoOptimization can still works, but the optimization schedule may be imprecise. This warning is maybe caused by the unstableness of your software resources (e.g., there may be other processes that sharing the same GPU or CPU).")
        else:
            for i in self.tbl._modules.keys():
                if not self.convergent[i]:
                    return
                assert not self.profiling[i]
            print_rank("All model profiles converge. ")
        self.schedule_optimization()

    def schedule_optimization(self):
        n = len(self.tbl._modules)
        profile_runtime = [self.tbl._modules[str(i)].profile.get_runtime() for i in range(n)]
        profile_memory = [self.tbl._modules[str(i)].profile.memory for i in range(n)]
        self._layer_runtime = profile_runtime
        self._layer_memory = profile_memory

        # Minimal Memory Test
        min_mem = ModelSimulator(
            layer_runtime = profile_runtime,
            layer_memory = profile_memory,
            layer_optim = [max_block_optim()] * n,
        ).simulate_train().peak_memory
        if min_mem > self.memory_limit:
            raise RuntimeError(f"Memory limit for TBL is {self.memory_limit} Byte, which exceeds the minimal possible memory for this TBL ({min_mem} Byte).")

        # Optimization
        print_rank(f"Scheduling auto optimization for TBL (algorithm name = {self.alg_name}) ... ")
        _start_time = time.time()
        best_optim_tensor, best_runtime = self.scheduling_algorithm(
            n = n,
            profile_runtime = profile_runtime,
            profile_memory = profile_memory,
            memory_limit = self.memory_limit,
            **self.kwargs
        )
        best_optim_tensor = optim_tensor_all_reduce(best_optim_tensor, best_runtime)
        if best_optim_tensor is not None:
            optim = _tensor2optim(best_optim_tensor)
        else:
            optim = [max_block_optim()] * n
        _time_cost = round(time.time() - _start_time, 2)

        model_sim = ModelSimulator(
            layer_runtime = profile_runtime,
            layer_memory = profile_memory,
            layer_optim = optim,
        ).simulate_train()
        print_rank("Estimated runtime =", model_sim.max_runtime, "sec. Estimated memory =", model_sim.peak_memory, "bytes.")
        self._estimated_runtime = model_sim.max_runtime
        self._estimated_memory = model_sim.peak_memory

        if _time_cost < 60:
            _time_cost = str(_time_cost) + "s"
        elif _time_cost < 60 * 60:
            _time_cost = str(round(_time_cost / 60, 2)) + "min"
        else:
            _time_cost = str(round(_time_cost / 3600, 2)) + "h"
        print_rank(f"Auto optimization for TBL is ready!! (time cost : {_time_cost})")
        self.apply_optimization(optim)
        self.save()

    def apply_optimization(self, optim):
        assert len(optim) == len(self.tbl._modules)
        for i in range(len(optim)):
            self.tbl._modules[str(i)].optimization = optim[i]
        self.optimize_scheduled = True
        self.optimization = optim

    def save(self, path = None):
        if config["rank"] == 0:
            if path is None:
                dir_name = ".tbl.optim.savings"
                os.makedirs(dir_name, exist_ok=True)
                path = os.path.join(dir_name, f"{time.time()}.json")
            dumped_obj = {
                "optim_tensor": _optim2tensor(self.optimization).tolist(),
                "optim": self.optimization,
                "alg_name": self.alg_name,
                "layer_runtime": self._layer_runtime,
                "layer_memory": self._layer_memory,
            }
            json.dump(dumped_obj, open(path, "w"))
            self.save_path = os.path.abspath(path)
            print_rank(f"Optimization file saved path: {path}")
        

class ModelSimulator:
    class Logger:
        def __init__(self, streams):
            self.streams = streams
            self.tasks = {s: [] for s in streams}
            self.stages = []
            self.stream2y = {s: len(streams) - i - 0.5 for i, s in enumerate(streams)}
            self.max_time = 0.
        
        def new_task(self, stream, start, end, name):
            self.tasks[stream].append({
                "start": start,
                "end": end,
                "duration": end - start,
                "name": name,
            })
            self.max_time = max(self.max_time, end)
        
        def new_stage(self, name, time):
            self.stages.append({
                "name": name,
                "time": time,
            })
        
        def gen_figure(self, path = None):
            import matplotlib.pyplot as plt

            figsize = (2 + 2.5 * max(map(len, self.tasks.values())), len(self.streams) + 1)
            fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot(111)
            plt.xticks(fontsize = 32)
            plt.yticks(list(self.stream2y.values()), list(self.stream2y.keys()), fontsize = 32)
            plt.xlim(0, self.max_time)
            plt.ylim(min(self.stream2y.values()) - 1, max(self.stream2y.values()) + 1)

            type2color = {
                "forward": "lightskyblue",
                "backward": "cornflowerblue",
                "scatter": "mediumorchid",
                "offload": "thistle",
                "prefetch": "lightgreen",
                "gather": "mediumseagreen",
                "optimizer": "gold",
            }

            for s in self.streams:
                for t in self.tasks[s]:
                    color = "silver"
                    for _type in type2color.keys():
                        if _type in t["name"]:
                            color = type2color[_type]
                            break
                    ax.add_patch(plt.Rectangle(
                        xy = (t["start"], self.stream2y[s] - 0.4),
                        width = t["duration"],
                        height = 0.8,
                        facecolor = color,
                        edgecolor = "black"
                    ))

            for s in self.stages:
                plt.plot([s["time"], s["time"]], [-1e9,1e9], linestyle = ":", color = "black", linewidth=2)

            if path is not None:
                fig.savefig(path, bbox_inches="tight")
            return fig


    def __init__(self, layer_runtime, layer_memory, layer_optim, auto_resident = True):
        assert len(layer_memory) == len(layer_runtime)
        self.n = len(layer_runtime)
        self.layer_runtime = layer_runtime
        self.layer_memory = layer_memory
        self.optim = layer_optim
        self.param_event = []
        self.hidden_event = []
        self.streams = {
            config["calc_stream"]: "calc_stream",
            config["load_stream"]: "load_stream",
        }
        if config["offload_stream"] not in self.streams:
            self.streams[config["offload_stream"]] = "offload_stream"
        if config["prefetch_stream"] not in self.streams:
            self.streams[config["prefetch_stream"]] = "prefetch_stream"
        for i in range(self.n):
            self.param_event.append(f"param_event_{i}")
            self.hidden_event.append(f"hidden_event_{i}")
        self.auto_resident = auto_resident
        self.clear()

    @property
    def layer_optim(self):
        return self.optim
    @layer_optim.setter
    def layer_optim(self, value):
        self.optim = value

    def clear(self):
        self.runtime = {}
        for s in self.streams.keys():
            self.runtime[s] = 0.
        for i in range(self.n):
            self.runtime[self.param_event[i]] = 0.
            self.runtime[self.hidden_event[i]] = 0.
        self.forwarding = False
        self.backwarding = False
        self.max_runtime = 0.

        if self.auto_resident:
            self.resident_memory = self.get_resident_memory()
        else:
            self.resident_memory = 0
        self.peak_memory = self.resident_memory
        self.delta_memory = 0
        self.memory_queue = []

        self.logger = ModelSimulator.Logger(list(self.streams.values()))

    def allocate_memory(self, mem, stream):
        if mem < 0:
            return self.deallocate_memory(-mem)
        if isinstance(stream, str):
            stream = config[stream]
        self.memory_queue.append((self.runtime[stream], mem))

    def deallocate_memory(self, mem, stream):
        if mem < 0:
            return self.allocate_memory(-mem)
        if isinstance(stream, str):
            stream = config[stream]
        self.memory_queue.append((self.runtime[stream], -mem))

    def wait(self, a, b):
        if isinstance(a, str) and "stream" in a:
            a = config[a]
        if isinstance(b, str) and "stream" in b:
            b = config[b]
        self.runtime[a] = max(self.runtime[a], self.runtime[b])

    def stream_synchronize(self):
        runtime = max(self.runtime.values())
        for s in self.runtime.keys():
            self.runtime[s] = runtime

        self.memory_queue.sort()
        for t, mem in self.memory_queue:
            self.delta_memory += mem
            self.peak_memory = max(self.delta_memory + self.resident_memory, self.peak_memory)
        self.memory_queue = []

    def stream_run(self, stream, runtime):
        if isinstance(stream, str):
            stream = config[stream]
        self.runtime[stream] += runtime
        self.max_runtime = max(self.max_runtime, self.runtime[stream])

    def new_task(self, stream_name, i, task_name):
        stream = config[stream_name]
        if stream_name not in self.streams.values():
            for name in self.streams.values():
                if config[name] == stream:
                    stream_name = name
                    break
        self.logger.new_task(
            stream = stream_name,
            start = self.runtime[stream],
            end = self.runtime[stream] + self.layer_runtime[i][task_name],
            name = task_name + "_" + str(i),
        )
        self.stream_run(stream, self.layer_runtime[i][task_name])
    
    def prefetch_parameter(self, i):
        if i < 0 or i >= self.n or i is None:
            return
        if not self.optim[i]["offload_parameter"]:
            return
        if self.optim[i]["zero_level"] == 2 and self.backwarding:
            return
        
        self.wait("prefetch_stream", "offload_stream")
        self.allocate_memory(self.layer_memory[i]["partitioned_parameter"], "prefetch_stream")
        self.new_task("prefetch_stream", i, "prefetch_parameter")
        self.wait(self.param_event[i], "prefetch_stream")

    def gather_parameter(self, i):
        if i < 0 or i >= self.n or i is None:
            return
        if self.optim[i]["zero_level"] == 2 and self.backwarding:
            return
        
        self.wait("load_stream", self.param_event[i])
        if not (self.backwarding and self.optim[i]["zero_level"] == 3 and not self.optim[i]["checkpointing"]):
            self.allocate_memory(self.layer_memory[i]["gathered_parameter"], "load_stream")
        self.new_task("load_stream", i, "gather_parameter")
        self.wait(self.param_event[i], "load_stream")

    def build_parameter(self, i):
        if i < 0 or i >= self.n or i is None:
            return

        self.wait("calc_stream", self.param_event[i])
        if self.backwarding:
            self.allocate_memory(self.layer_memory[i]["gathered_gradient"], "calc_stream")

    def scatter_gradient(self, i):
        if i < 0 or i >= self.n or i is None:
            return

        if self.backwarding:
            self.wait("load_stream", "calc_stream")
            self.allocate_memory(self.layer_memory[i]["partitioned_gradient"], "load_stream")
            self.new_task("load_stream", i, "scatter_gradient")
            self.wait(self.param_event[i], "load_stream")
            self.deallocate_memory(self.layer_memory[i]["gathered_gradient"], "load_stream")
        if self.backwarding or (self.optim[i]["zero_level"] == 3 and self.optim[i]["checkpointing"]):
            self.deallocate_memory(self.layer_memory[i]["gathered_parameter"], "load_stream")

    def offload_gradient(self, i):
        if i < 0 or i >= self.n or i is None:
            return

        if self.backwarding:
            self.wait("offload_stream", "prefetch_stream")
            self.wait("offload_stream", self.param_event[i])
            self.new_task("offload_stream", i, "offload_gradient")
            self.deallocate_memory(self.layer_memory[i]["partitioned_gradient"], "offload_stream")
            self.wait(self.param_event[i], "offload_stream")
        if (self.backwarding or self.optim[i]["zero_level"] == 3) and self.optim[i]["offload_parameter"]:
            self.deallocate_memory(self.layer_memory[i]["partitioned_parameter"], "offload_stream")


    def prefetch_hidden_state(self, i):
        if i < 0 or i >= self.n or i is None:
            return
        if not self.optim[i]["offload_hidden_state"]:
            return
        
        self.wait("prefetch_stream", "offload_stream")
        self.allocate_memory(self.layer_memory[i]["input_tensor"], "prefetch_stream")
        self.new_task("prefetch_stream", i, "prefetch_hidden_state")
        self.wait(self.hidden_event[i], "prefetch_stream")
        self.wait("offload_stream", "prefetch_stream")


    def offload_hidden_state(self, i):
        if i < 0 or i >= self.n or i is None:
            return
        if not self.optim[i]["offload_hidden_state"]:
            self.allocate_memory(self.layer_memory[i]["input_tensor"], "offload_stream")
            return
        
        self.wait("offload_stream", "prefetch_stream")
        self.new_task("offload_stream", i, "offload_hidden_state")
        self.wait(self.hidden_event[i], "offload_stream")
        self.wait("prefetch_stream", "offload_stream")

    def calculate(self, i, key):
        if i < 0 or i >= self.n or i is None:
            return

        for k in self.runtime.keys():
            self.wait(k, "calc_stream")
        self.allocate_memory(self.layer_memory[i][f"{key}_peak"], "calc_stream")
        self.new_task("calc_stream", i, key)
        self.deallocate_memory(self.layer_memory[i][f"{key}_peak"] - self.layer_memory[i][f"{key}_delta"], "calc_stream")

    def forward_no_grad(self, i):
        self.calculate(i, "forward_no_grad")

    def forward_with_grad(self, i):
        self.calculate(i, "forward_with_grad")

    def backward(self, i):
        self.calculate(i, "backward")

    def simulate_forward(self, i = None, clear = True, head = False, tail = False):
        if clear:
            self.clear()
        self.forwarding = True
        self.backwarding = False
        if i is None:
            i = range(self.n)
        if isinstance(i, Iterable):
            _i = i
            self.stream_synchronize()
            self.simulate_forward(i = _i[0], clear = False, head = True)
            for i in _i:
                self.simulate_forward(i, clear = False)
            self.simulate_forward(i = _i[-1], clear = False, tail = True)
            self.stream_synchronize()
            return self
        
        if head:
            self.prefetch_parameter(i)
            self.gather_parameter(i)
            return self
        if tail:
            self.scatter_gradient(i)
            self.offload_gradient(i)
            return self
        if self.optim[i]["segment_synchronization"]:
            self.stream_synchronize()
        self.scatter_gradient(i-1)
        self.offload_gradient(i-1)
        self.wait("prefetch_stream", "calc_stream")
        self.wait("load_stream", "calc_stream")
        if not self.optim[i]["economical_forward"]:
            self.prefetch_parameter(i+1)
            self.gather_parameter(i+1)
        self.offload_hidden_state(i)
        if self.optim[i]["economical_forward"]:
            self.prefetch_parameter(i+1)
            self.gather_parameter(i+1)
        self.build_parameter(i)
        if self.optim[i]["checkpointing"]:
            self.forward_no_grad(i)
        else:
            self.forward_with_grad(i)

        return self

    def simulate_backward(self, i = None, clear = True, head = False, tail = False):
        if clear:
            self.clear()
        self.forwarding = False
        self.backwarding = True
        if i is None:
            i = range(self.n)
        if isinstance(i, Iterable):
            _i = i
            self.stream_synchronize()
            self.simulate_backward(i = _i[-1], clear = False, head = True)
            for i in reversed(_i):
                self.simulate_backward(i, clear = False)
            self.simulate_backward(i = _i[0], clear = False, tail = True)
            self.stream_synchronize()
            return self
        
        if head:
            self.prefetch_parameter(i)
            self.gather_parameter(i)
            self.prefetch_hidden_state(i)
            return self
        if tail:
            self.scatter_gradient(i)
            self.offload_gradient(i)
            return self
        if self.optim[i]["segment_synchronization"]:
            self.stream_synchronize()

        self.wait("prefetch_stream", "calc_stream")

        if config["nvlink_available"]:
            self.scatter_gradient(i+1)
            self.prefetch_parameter(i-1)
            if not self.optim[i]["economical_backward"]:
                self.prefetch_hidden_state(i-1)
                self.gather_parameter(i-1)
                self.offload_gradient(i+1)
        else:
            self.scatter_gradient(i+1)
            self.offload_gradient(i+1)
            if not self.optim[i]["economical_backward"]:
                self.prefetch_hidden_state(i-1)
                self.prefetch_parameter(i-1)
                self.gather_parameter(i-1)

        self.build_parameter(i)
        self.wait("calc_stream", self.hidden_event[i])
        if self.optim[i]["checkpointing"]:
            self.forward_with_grad(i)
        self.backward(i)

        if config["nvlink_available"]:
            if self.optim[i]["economical_backward"]:
                self.gather_parameter(i-1)
                self.offload_gradient(i+1)
                self.prefetch_hidden_state(i-1)
        else:
            if self.optim[i]["economical_backward"]:
                self.prefetch_hidden_state(i-1)
                self.prefetch_parameter(i-1)
                self.gather_parameter(i-1)



        return self

    def simulate_optimizer(self, i = None, clear = True):
        if clear:
            self.clear()
        self.forwarding = False
        self.backwarding = False
        self.stream_synchronize()
        if i is None:
            for i in range(self.n):
                self.new_task("calc_stream", i, "optimizer_step")
                if not self.optim[i]["offload_parameter"]:
                    self.wait("prefetch_stream", "calc_stream")
                    self.new_task("prefetch_stream", i, "prefetch_parameter")
        else:
            if i > 0:
                if not self.optim[i - 1]["offload_parameter"]:
                    self.wait("prefetch_stream", "calc_stream")
                    self.new_task("prefetch_stream", i - 1, "prefetch_parameter")
            self.new_task("calc_stream", i, "optimizer_step")
            if i == self.n - 1:
                if not self.optim[i]["offload_parameter"]:
                    self.wait("prefetch_stream", "calc_stream")
                    self.new_task("prefetch_stream", i, "prefetch_parameter")
    
        self.stream_synchronize()
        return self

    def get_resident_memory(self, i = None):
        if i is None:
            i = range(self.n)
        if isinstance(i, Iterable):
            _i = i
            ret = 0
            for i in _i:
                ret += self.get_resident_memory(i)
            return ret
        
        if self.optim[i]["offload_parameter"]:
            return 0
        return self.layer_memory[i]["partitioned_parameter"]

    def simulate_train(self):
        self.clear()
        self.logger.new_stage("forward", 0.)
        self.simulate_forward(clear = False)
        self.logger.new_stage("backward", self.max_runtime)
        self.simulate_backward(clear = False)
        self.logger.new_stage("optimize", self.max_runtime)
        self.simulate_optimizer(clear = False)
        return self


