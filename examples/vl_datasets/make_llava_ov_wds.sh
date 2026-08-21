export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export CUDA_LAUNCH_BLOCKING=1

LLaVA-NeXT-HOME="Path_Of_LLaVA-NeXT"
VISION_MODEL_PATH="Path_Of_VISION_MODEL"
PROMPT_VERSION="qwen_1_5"

# stage1
image_aspect_ratio=square

# other stages
image_aspect_ratio=anyres_max_9

set -u
  DATA_PATH=$1
  EXPNAME_PATH=$2
  HOSTFILE=$3
set +u

echo "BASE_RUN_NAME: ${EXPNAME_PATH}"

CKPT_PATH="./checkpoints"

mkdir -p $CKPT_PATH
mkdir -p $EXPNAME_PATH
LOGFILE=$EXPNAME_PATH/exp.log
i=0
NNodes=`wc -l ${HOSTFILE} | cut -d " " -f1`
MASTER_ADDR=`head -n 1 ${HOSTFILE} | cut -d " " -f1`
echo "Master node: ${MASTER_ADDR}"
echo ${NNodes}
echo ${i}
echo ${MASTER_ADDR}

for ip in `cat ${HOSTFILE} | cut -d " " -f1`
do
    echo "Starting node ${i}/${NNodes}: ${ip}"
    ssh $ip \
    "cd ${PWD} && \
    export WANDB_MODE=offline && \
    export ACCELERATE_CPU_AFFINITY=1 && \
    export PYTHONPATH=$LLaVA-NeXT-HOME:$PYTHONPATH && \
    torchrun --nproc_per_node=4 --nnodes=${NNodes} --node_rank=${i} --master_addr=${MASTER_ADDR} --master_port=29513 llava_ov_wds.py \
        --model_name_or_path ${CKPT_PATH} \
        --version ${PROMPT_VERSION} \
        --data_path $DATA_PATH \
        --image_folder playground/data \
        --video_folder ./onevision_data/videos \
        --mm_tunable_parts="mm_mlp_adapter" \
        --mm_vision_tower_lr=2e-6 \
        --vision_tower ${VISION_MODEL_PATH} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --mm_spatial_pool_mode "bilinear" \
        --group_by_modality_length True \
        --image_aspect_ratio ${image_aspect_ratio} \
        --image_grid_pinpoints '(1x1),...,(6x6)' \
        --mm_patch_merge_type spatial_unpad \
        --bf16 True \
        --run_name $EXPNAME_PATH \
        --output_dir "${EXPNAME_PATH}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 20 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 2 \
        --lazy_preprocess True \
        --torch_compile True \
        --torch_compile_backend "inductor" \
        --dataloader_drop_last True \
        --seed 42 \
        --do_train False \
        --frames_upbound 32 1>>$LOGFILE.$ip 2>&1" &
    i=`expr $i + 1`
done
