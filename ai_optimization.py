import os
import sys
import json
import subprocess
import collections
import socket
from flagai.logger import log_dist

import signal
import numpy as np
import tensorflow as tf  # For AI features like dynamic resource allocation and advanced logging

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
                log_dist(f"Error parsing hostfile line: {line}. Error: {err}")
                continue  # Skip invalid lines
            if hostname in resource_pool:
                log_dist(f"Warning: host {hostname} is already defined in the hostfile.")
                continue
            resource_pool[hostname] = slot_count

    return resource_pool

def cmd_load_hyperparam(config_path=None, format="json", encoding="utf-8"):
    """
    Load arguments from argparse and config file
    """
    original_format = format  # Store original format
    if not config_path:
        raise ValueError("Configuration path must be provided.")
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    format = config_path.rsplit('.')[-1]
    with open(config_path, 'r', encoding=encoding) as f:
        if format == "json":
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported format {format} for hyperparam file. Only JSON is supported.")
    
    config_cmd = []
    for key, value in config_dict.items():
        if value:
            config_cmd.append(f'--{key}')
            config_cmd.append(str(value))
        else:
            config_cmd.append(f'--{key}')
    return config_cmd

def optimize_resource_allocation(resource_pool):
    """
    AI-driven resource allocation based on current workload and system state.
    """
    rng = np.random.default_rng()  # Use the new numpy random generator
    optimized_resources = {}
    for host, slots in resource_pool.items():
        # Example of a dummy optimization process
        optimized_resources[host] = int(slots * rng.uniform(0.8, 1.2))
    return optimized_resources

def analyze_logs(log_file):
    """
    Analyze logs using AI to detect patterns or anomalies.
    """
    if not os.path.isfile(log_file):
        log_dist(f"Log file not found at: {log_file}")
        return

    # Placeholder for AI-based log analysis
    # e.g., using TensorFlow or custom algorithms to analyze and interpret log data
    with open(log_file, 'r') as f:
        logs = f.read()
    
    # Example of dummy log analysis
    if "error" in logs:
        log_dist("Potential issues detected in logs. Review necessary.")

def launch_dist(launcher='distributed_deepspeed',
                num_nodes=1,
                gpus_per_node=1,
                master_addr='localhost',
                master_port=17500,
                hostfile='hostfile',
                nccl_info=False,
                training_script='train.py',
                training_script_paras=None,
                training_paras=None):
    try:
        resource_pool = fetch_hostfile(hostfile)
        if resource_pool:
            resource_pool = optimize_resource_allocation(resource_pool)
    except Exception as e:
        log_dist(f"Error occurred: {e}")
        raise RuntimeError("Failed during resource allocation or fetching hostfile")

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
        launch_cmd(run_cmd)

    elif num_nodes == 1 and launcher == 'simple_torch':
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
            launch_cmd(run_cmd)
    else:
        raise ValueError('No available launcher')

    # Post-execution log analysis
    log_file = '/path/to/log/file.log'  # Update with the actual log file path
    analyze_logs(log_file)
