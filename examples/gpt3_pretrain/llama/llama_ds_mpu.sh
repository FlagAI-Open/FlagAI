GPU_NUM_PER_NODE=2

## Trigger job only on Master Node when ds+mpu.
### model_parallel_size should be set as needed.
deepspeed --num_gpus $GPU_NUM_PER_NODE --num_nodes $WORLD_SIZE --hostfile hostfile.2n2g --master_port $MASTER_PORT llama_ds_mpu.py --not_call_launch

## Trigger job on Each Node when bmt or ddp.
### Env Var should be set.
python -m torch.distributed.launch --nproc_per_node $GPU_NUM_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT train_llama_bmtrain.py --not_call_launch
