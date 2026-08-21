# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# Copyright 2020 The Microsoft DeepSpeed Team
"""
FlagAI runner is the main front-end to launching multi-worker
training jobs with DeepSpeed. By default this uses pdsh to parallel
ssh into multiple worker nodes and launch all the necessary processes
per rank for training.
"""

import os
import sys
import json
import subprocess
import collections
import socket
from flagai.logger import log_dist

import signal

def launch_cmd(run_cmd):
    p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
    def signal_handler(signal, frame):
        os.killpg(os.getpgid(p.pid), 9)
    signal.signal(signal.SIGINT, signal_handler)
    p.wait()

def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        log_dist("Unable to find hostfile, will proceed with training "
                 "with local resources only.")
        return None
    # e.g., worker-0 slots=16
    with open(hostfile_path, 'r') as fd:
        resource_pool = collections.OrderedDict()
        for line in fd.readlines():
            line = line.strip()
            if line == '':
                # skip empty lines
                continue
            try:
                hostname, slots = line.split()
                _, slot_count = slots.split("=")
                slot_count = int(slot_count)
            except ValueError as err:
                raise err
            if hostname in resource_pool:
                raise ValueError(f"host {hostname} is already defined")
            resource_pool[hostname] = slot_count

    return resource_pool


def cmd_load_hyperparam(config_path=None, format="json", encoding="utf-8"):
    """
    shell load arguments form argparse and config file
    """
    # config_path='config/config_block_large_chinese.json'
    format = config_path.rsplit('.')[-1]
    with open(config_path, 'r', encoding=encoding) as f:
        if format == "json":
            config_dict = json.load(f)
        else:
            raise NameError("current format%s for hyperparam file is invalid" %
                            format)
    config_cmd = []
    for key in config_dict:
        if len(str(config_dict[key])) == 0:
            config_cmd.append('--' + key)
        else:
            config_cmd.append('--' + key)
            config_cmd.append(str(config_dict[key]))
    return config_cmd


def launch_dist(launcher='distributed_deepspeed',
                num_nodes=1,
                gpus_per_node=1,
                master_addr='localhost',
                master_port=17500,
                hostfile='hostfile',
                nccl_info=False,
                training_script='train.py',
                training_script_paras=None,
                training_paras=None,):
    try:
        resource_pool = fetch_hostfile(hostfile)
    except:
        raise RuntimeError("hostfile is not valid")
    # respect CUDA_VISIBLE_DEVICES for a single node and no explicit resource filters
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    if num_nodes > 1 and launcher == 'distributed_torch':
        node_rank = 0
        for host, slots in resource_pool.items():
            cmd_launch = ['pdsh', '-f', '1024', '-w']
            cmd_launch.append('ssh:' + host)
            cmd_launch.append('"')
            if nccl_info:
                cmd_launch.extend([
                    'export NCCL_DEBUG=info;', 'export NCCL_IB_DISABLE=0;',
                    'export NCCL_NET_GDR_LEVEL=2;'
                ])
            cmd_launch.extend([
                'export NUM_NODES=' + str(num_nodes) + ';',
                'export GPUS_PER_NODE=' + str(gpus_per_node) + ';',
                'export NCCL_NET_GDR_LEVEL=2;', sys.executable, '-m',
                'torch.distributed.launch'
            ])
            torch_distributed_args = [
                '--nproc_per_node',
                str(gpus_per_node),
                '--nnodes',
                str(num_nodes),
                '--node_rank',
                str(node_rank),
                '--master_addr',
                master_addr,
                '--master_port',
                str(master_port),
            ]
            cmd_launch.extend(torch_distributed_args)
            cmd_launch.append(training_script)

            for para in training_script_paras:
                if 'training_script_config' in para:
                    para_index = training_script_paras.index(para)
                    training_script_args = cmd_load_hyperparam(
                        training_script_paras[para_index + 1])
                    cmd_launch.extend(training_script_args)
                    del training_script_paras[para_index:para_index + 2]
            if len(training_script_paras) > 0:
                cmd_launch.extend(training_script_paras)
            cmd_launch.append('--not_call_launch')
            cmd_launch.append('"')
            run_cmd = ' '.join(cmd_launch)
            log_dist(run_cmd)
            p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
            def signal_handler(signal, frame):
                os.killpg(os.getpgid(p.pid), 9)
            signal.signal(signal.SIGINT, signal_handler)
            p.wait()
            # subprocess.Popen(run_cmd, shell=True)
            node_rank += 1

    elif num_nodes == 1 and launcher == 'distributed_torch':
        cmd_launch = []
        cmd_launch.extend([
            'export NUM_NODES=' + str(num_nodes) + ';',
            'export GPUS_PER_NODE=' + str(gpus_per_node) + ';', sys.executable,
            '-m', 'torch.distributed.launch'
        ])
        torch_distributed_args = [
            '--nproc_per_node',
            str(gpus_per_node),
            '--nnodes',
            str(num_nodes),
            '--node_rank',
            str(0),
            '--master_addr',
            master_addr,
            '--master_port',
            str(master_port),
        ]
        cmd_launch.extend(torch_distributed_args)
        cmd_launch.append(training_script)
        if training_paras:
            cmd_launch.extend(training_paras)

        cmd_launch.append('--not_call_launch')
        run_cmd = ' '.join(cmd_launch)
        log_dist(run_cmd)
        # subprocess.Popen(run_cmd, shell=True)

        launch_cmd(run_cmd)

    elif launcher == 'distributed_deepspeed':
        if hostfile is None:
            log_dist(
                'Unable to find hostfile, will proceed with training with local resources only.'
            )

            with open('/tmp/hostfile', 'w') as w:
                w.write(socket.gethostname() + ' slots=2')
            hostfile = '/tmp/hostfile'

        if nccl_info:
            cmd_launch = [
                'NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed'
            ]
        else:
            cmd_launch = ['deepspeed']

        cmd_launch.extend([
            '--master_port',
            str(master_port),
            '--num_nodes',
            str(num_nodes),
            '--num_gpus',
            str(gpus_per_node),
            '--hostfile',
            hostfile,
        ])

        cmd_launch.append(training_script)
        if training_script_paras:
            for para in training_script_paras:
                if 'training_script_config' in para:
                    para_index = training_script_paras.index(para)
                    training_script_args = cmd_load_hyperparam(
                        training_script_paras[para_index + 1])
                    cmd_launch.extend(training_script_args)
                    del training_script_paras[para_index:para_index + 2]
            if len(training_script_paras) > 0:
                cmd_launch.extend(training_script_paras)

        if training_paras:
            cmd_launch.extend(training_paras)

        cmd_launch.append('--not_call_launch')
        run_cmd = ' '.join(cmd_launch)
        log_dist(run_cmd)
        # subprocess.Popen(run_cmd, shell=True)
        launch_cmd(run_cmd)


    elif num_nodes == 1 and launcher == 'simple_torch':
        # This launcher
        for gpu_id in range(gpus_per_node):
            cmd_launch = []
            cmd_launch.extend([
                'export MASTER_ADDR=' + str(master_addr) + ';',
                'export MASTER_PORT=' + str(master_port) + ';', sys.executable
            ])
            cmd_launch.append(training_script)
            torch_distributed_args = [
                '--gpu_nums',
                str(gpus_per_node), '--local_rank',
                str(gpu_id)
            ]
            cmd_launch.extend(torch_distributed_args)
            for para in training_script_paras:
                if 'training_script_config' in para:
                    para_index = training_script_paras.index(para)
                    training_script_args = cmd_load_hyperparam(
                        training_script_paras[para_index + 1])
                    cmd_launch.extend(training_script_args)
                    del training_script_paras[para_index:para_index + 2]
            if len(training_script_paras) > 0:
                cmd_launch.extend(training_script_paras)

            if training_paras:
                cmd_launch.extend(training_paras)

            run_cmd = ' '.join(cmd_launch)
            log_dist(run_cmd)
            # subprocess.Popen(run_cmd, shell=True)
            launch_cmd(run_cmd)
    else:
        raise Exception('No aviable launcher')
