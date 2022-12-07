# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for logging and serialization"""
import os
import random
import time
import numpy as np
import torch
import subprocess
from flagai import mpu
from flagai.logger import log_dist
import torch.distributed as dist


def get_hostname():
    hostname_cmd = ["hostname -I"]
    result = subprocess.check_output(hostname_cmd, shell=True)
    master_addr = result.decode('utf-8').split()[0]
    return master_addr


def get_spare_port(args):
    if torch.distributed.get_rank() == 0:
        port = subprocess.check_output(["shuf -n 1 -i 10000-65535"],
                                       shell=True)
        port = int(port.strip())
        if port == args.master_port:
            port = subprocess.check_output(["shuf -n 1 -i 10000-65535"],
                                           shell=True)
            port = int(port.strip())
        port = torch.cuda.LongTensor([port])
    else:
        port = torch.cuda.LongTensor([0])
    torch.distributed.broadcast(port, 0)
    port = port.item()
    return port


class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            #torch.cuda.synchronize() #TODO is nessisary
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            #torch.cuda.synchronize() #TODO change here
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}
        self.verbose = True

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a train_args of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if self.verbose:
            log_dist(string, ranks=[0])


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() /
                                        mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() /
                                            mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(torch.cuda.memory_reserved() /
                                         mega_bytes)
    log_dist(string)


def get_checkpoint_name(checkpoints_path, iteration):
    # TODO 根据不同的env_type来设置
    iteration = int(iteration)
    d = '{:d}'.format(iteration)
    env_type = os.getenv("ENV_TYPE")
    if env_type == "deepspeed+mpu":
        filename = 'pytorch_model_{:02d}.bin'.format(
            mpu.get_model_parallel_rank())
    else:
        filename = 'pytorch_model.bin'
    return os.path.join(checkpoints_path, d, filename)


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_iteration.txt')


def save_checkpoint(iteration,
                    best_iteration,
                    model,
                    optimizer,
                    lr_scheduler,
                    barrier=True,
                    save_dir='checkpoints',
                    only_changed_parameters=False,
                    save_optim=True,
                    save_rng=True):
    """Save a model checkpoint."""
    env_type = os.getenv('ENV_TYPE')
    # Only rank zer0 of the data parallel writes to the disk.
    checkpoint_name = get_checkpoint_name(save_dir, str(iteration))
    log_dist(
        'global rank {} is saving checkpoint at iteration {:7d} to {}'.format(
            0, iteration, checkpoint_name))
    sd = {'iteration': iteration}

    # model state_dict
    while hasattr(model, 'module'): 
        model = model.module
    # if only_changed_parameters:
    #     requires_grad_dict = {}
    #     for name, parameter in model.named_parameters():
    #         requires_grad_dict[name] = parameter.requires_grad
    #     state_dict = {
    #         key: value
    #         for key, value in state_dict.items() if requires_grad_dict[key]
    #     }
    # else:
    state_dict = model.state_dict()
    sd['module'] = state_dict

    # Optimizer stuff.
    if save_optim:
        if optimizer is not None:
            sd['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            sd['lr_scheduler'] = lr_scheduler.state_dict()
    # rng states.
    if save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
    if env_type == 'pytorch' or (env_type != 'deepspeed+mpu' and env_type != 'bmtrain'
                                 and dist.get_rank() == 0) or (
                                    env_type == 'deepspeed+mpu'and mpu.get_model_parallel_src_rank() == 0):
        ensure_directory_exists(checkpoint_name)
        config_path = os.path.join(save_dir, str(iteration), 'config.json')

        if hasattr(model, 'save_config'):
            model.save_config(config_path)
            log_dist('  successfully saved {}'.format(config_path))
        torch.save(sd, checkpoint_name)
        log_dist('  successfully saved {}'.format(checkpoint_name))

        tracker_filename = get_checkpoint_tracker_filename(save_dir)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration) + '\t' + str(best_iteration))
    elif  env_type == 'bmtrain' and os.environ.get('RANK') == 0:
        ensure_directory_exists(checkpoint_name)
        config_path = os.path.join(save_dir, str(iteration), 'config.json')

        if hasattr(model, 'save_config'):
            model.save_config(config_path)
            log_dist('  successfully saved {}'.format(config_path))
        torch.save(sd, checkpoint_name)
        log_dist('  successfully saved {}'.format(checkpoint_name))

        tracker_filename = get_checkpoint_tracker_filename(save_dir)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration) + '\t' + str(best_iteration))

    # Wait so everyone is done (necessary)
    if barrier and dist.is_initialized():
        torch.distributed.barrier()
    # And update the latest iteration


def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        log_dist('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        if os.path.isdir(load_path):
            path = os.path.normpath(load_path)
            load_dir, iteration = os.path.split(path)
            log_dist('Try to directly load the checkpoint from the directory')
            return load_dir, iteration, -1, True
    else:
        log_dist('read the metadata file {} '.format(tracker_filename))
        with open(tracker_filename, 'r', encoding='utf8') as infile:
            iteration, best_iteration = infile.readline().strip().split('\t')
        return load_path, iteration, best_iteration, True
    log_dist('    will not load any checkpoints and will start from '
             'random')
    return load_path, -1, -1, False


def load_checkpoint(model, load_dir="checkpoints", load_type='latest'):
    """Load a model checkpoint."""

    load_dir, iteration, best_iteration, success = get_checkpoint_iteration(
        load_dir)

    if not success:
        return 0

    # Checkpoint.
    if load_type == 'latest':
        checkpoint_name = get_checkpoint_name(load_dir, iteration)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, best_iteration)

    log_dist('global rank {} is loading checkpoint {}'.format(
        0, checkpoint_name))
    sd = torch.load(checkpoint_name, map_location='cpu')

    while hasattr(model, 'module'):
        model = model.module
    model.load_state_dict(sd['module'], strict=False)
    del sd['module']
    return sd


def load_optim(optimizer, lr_scheduler, sd):
    # Optimizer.
    try:
        if optimizer is not None:
            optimizer.load_state_dict(sd['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(sd['lr_scheduler'])
        log_dist('global rank 0 is loading optimizer & lr_scheduler')
    except KeyError:
        log_dist('Unable to load optimizer from checkpoint, exiting. '
                 'Specify --no-load-optim or --finetune to prevent '
                 'attempting to load the optimizer '
                 'state.')


def load_rng(sd):
    # rng states.
    env_type = os.getenv('ENV_TYPE')
    try:
        random.setstate(sd['random_rng_state'])
        np.random.set_state(sd['np_rng_state'])
        torch.set_rng_state(sd['torch_rng_state'])
        torch.cuda.set_rng_state(sd['cuda_rng_state'])
        if env_type == 'deepspeed+mpu':
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        log_dist('global rank 0 is loading rng states')
    except KeyError:
        log_dist('Unable to load random state from checkpoint, exiting. '
                 'Specify --no-load-rng or --finetune to prevent '
                 'attempting to load the random '
                 'state.')
    log_dist('  successfully loaded rng checkpoints')
