#!/bin/bash
#SBATCH --job-name=cpm2-test
#SBATCH --partition=rtx2080

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8


MASTER_PORT=30123
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# load python virtualenv if you have
# source /path/to/python/virtualenv/bin/activate
 
# uncomment to print nccl debug info
# export NCCL_DEBUG=info

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_HOST:$MASTER_PORT train.py


