python -m launch --launcher distributed_deepspeed \
       --hostfile ./hostfile \
       --master_addr 172.31.255.4\
       --master_port 17885 \
       --gpus_per_node 2 \
       --num_nodes 1 \
       ./train_t5.py \
       --deepspeed_config ./deepspeed.json \
       --model_parallel_size 1 

