export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4

python train_multi_gpu.py