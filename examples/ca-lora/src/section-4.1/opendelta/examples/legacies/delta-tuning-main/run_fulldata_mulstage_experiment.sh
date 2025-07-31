# Required environment variables:
# TAG: tag for the trail
# PROMPT: whether to use prompt (prompt or none)
# TASK: SST-2 / CoLA / MNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list
# HARD: whether to use hard template (Y or N)
# NOTRAIN: not to train
# GPUN: number of gpu

# Number of training instances per label

# Training steps
MAX_STEP=6000

# Validation steps
EVAL_STEP=200

# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
TASK_EXTRA=""
case $TASK in
    CoLA)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent_0*_This_is*mask*.*sep+*
                ;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**mask**sep+*
                ;;
        esac
        MAPPING="{'0':'incorrect','1':'correct'}"
        ;;
    SST-2)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent_0*_It_was*mask*.*sep+*;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**mask**sep+*;;
        esac
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    MRPC)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent_0**mask*,*+sentl_1**sep+*;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**sent_1**sep+**mask**sep+*;;
        esac
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QQP)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent_0**mask*,*+sentl_1**sep+*;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**sent_1**sep+**mask**sep+*;;
        esac
        MAPPING="{'0':'No','1':'Yes'}"
        TASK_EXTRA="--num_sample 4"
        ;;
    STS-B)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent_0**mask*,*+sentl_1**sep+*;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**sent_1**sep+**mask**sep+*;;
        esac
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    MNLI)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent-_0*?*mask*,*+sentl_1**sep+*;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**sent_1**sep+**mask**sep+*;;
        esac
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 512 --num_sample 4"
        ;;
    QNLI)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent-_0*?*mask*,*+sentl_1**sep+*;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**sent_1**sep+**mask**sep+*;;
        esac
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 512"
        ;;
    RTE)
        case $HARD in
            Y)
                TEMPLATE=*cls*$PROMPT*sent-_0*?*mask*,*+sentl_1**sep+*;;
            N)
                TEMPLATE=*cls*$PROMPT*sent_0**sep+**sent_1**sep+**mask**sep+*;;
        esac
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 512 --first_sent_limit 240"
        ;;
esac

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
PER_BS=$(expr $BS / $GPUN)
REAL_BS=32
GS=$(expr ${PER_BS} / ${REAL_BS})

# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM
DATA_DIR=data/original/$TASK

if [ -z "$NOTRAIN" ]; then
    python run.py \
    --task_name $TASK \
    --data_dir $DATA_DIR \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --model_name_or_path $MODEL \
    --prompt_num 10 \
    --max_seq_length 256 \
    --per_device_train_batch_size $REAL_BS \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GS \
    --learning_rate $LR \
    --max_steps $MAX_STEP \
    --logging_steps $EVAL_STEP \
    --eval_steps $EVAL_STEP \
    --num_train_epochs 0 \
    --output_dir result/$TASK-full-mul-$PROMPT-$SEED-$MODEL-$TRIAL_IDTF \
    --seed $SEED \
    --tag $TAG \
    --template $TEMPLATE \
    --mapping $MAPPING \
    $TASK_EXTRA \
    $1
else
    python run.py \
    --task_name $TASK \
    --data_dir $DATA_DIR \
    --overwrite_output_dir \
    --do_eval \
    --evaluation_strategy steps \
    --model_name_or_path $MODEL \
    --prompt_num 10 \
    --max_seq_length 256 \
    --per_device_train_batch_size $REAL_BS \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GS \
    --max_steps $MAX_STEP \
    --logging_steps $EVAL_STEP \
    --eval_steps $EVAL_STEP \
    --num_train_epochs 0 \
    --output_dir result/$TASK-full-mul-$PROMPT-$SEED-$MODEL-$TRIAL_IDTF \
    --seed $SEED \
    --tag $TAG \
    --template $TEMPLATE \
    --mapping $MAPPING \
    $TASK_EXTRA \
    $1
fi
if (($? != 0)); then exit 1; fi

# Delete the checkpoint
# Since we need to run multiple trials, saving all the checkpoints takes
# a lot of storage space. You can find all evaluation results in `log` file anyway.
# rm -r result/$TASK-full-mul-$PROMPT-$SEED-$MODEL-$TRIAL_IDTF
