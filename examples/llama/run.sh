export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4
mkdir current_training
cp train_llama.py current_training
cp deepspeed.json current_training
python -u train_llama.py > current_training/train_30b_llama_node16_mp4_bs4_zero2_vocab100000_alldatav1.log 2>&1
