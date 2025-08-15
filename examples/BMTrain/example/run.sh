export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py
