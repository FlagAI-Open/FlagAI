## Trigger job only on Master Node when ds+mpu
deepspeed --num_gpus 2 --num_nodes $WORLD_SIZE --hostfile hostfile.2n2g --master_port $MASTER_PORT llama_ds_mpu.py --not_call_launch

## Trigger job on Each Node when bmt or ddp
python -m torch.distributed.launch --nproc_per_node 2 --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT train_llama_bmtrain.py --not_call_launch
