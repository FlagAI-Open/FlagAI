# Required environment variables:
# TASK: SST-2 / CoLA / MNLI / QNLI / RTE / MRPC / QQP / STS-B

TAG=exp-stage
model=roberta-large
cuda=1,2
bs=4
gpun=2

mkdir ./result/$TASK

case $TASK in
    CoLA)
        task=cola
    ;;
    SST-2)
        task=sst-2
    ;;
    MNLI)
        task=mnli
    ;;
    STS-B)
        task=sts-b/pearson
    ;;
    MRPC)
        task=mrpc/f1
    ;;
    QQP)
        task=qqp/f1
    ;;
    QNLI)
        task=qnli
    ;;
    RTE)
        task=rte
    ;;
esac

for hard in Y N
do

    # PROMPT ADAPTER BIAS
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3 1e-3 3e-4
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG$hard \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            TRAIN_PARAM='prompt adapter bias' \
            bash run_mulstage_experiment.sh "--use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG$hard', 'task_name': '$task', 'training_params': ['prompt', 'adapter', 'bias']}" > ./result/$TASK/$TASK-$hard-stage-prompt-adapter-bias.out

    # PROMPT BIAS ADAPTER
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3 1e-3 3e-4
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG$hard \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            TRAIN_PARAM='prompt bias adapter' \
            bash run_mulstage_experiment.sh "--use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG$hard', 'task_name': '$task', 'training_params': ['prompt', 'bias', 'adapter']}" > ./result/$TASK/$TASK-$hard-stage-prompt-bias-adapter.out

    # BIAS PROMPT ADAPTER
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3 1e-3 3e-4
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG$hard \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            TRAIN_PARAM='bias prompt adapter' \
            bash run_mulstage_experiment.sh "--use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG$hard', 'task_name': '$task', 'training_params': ['bias', 'prompt', 'adapter']}" > ./result/$TASK/$TASK-$hard-stage-bias-prompt-adapter.out

    # BIAS ADAPTER PROMPT
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3 1e-3 3e-4
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG$hard \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            TRAIN_PARAM='bias adapter prompt' \
            bash run_mulstage_experiment.sh "--use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG$hard', 'task_name': '$task', 'training_params': ['bias', 'adapter', 'prompt']}" > ./result/$TASK/$TASK-$hard-stage-bias-prompt-adapter.out

    # ADAPTER PROMPT BIAS
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3 1e-3 3e-4
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG$hard \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            TRAIN_PARAM='adapter prompt bias' \
            bash run_mulstage_experiment.sh "--use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG$hard', 'task_name': '$task', 'training_params': ['adapter', 'prompt', 'bias']}" > ./result/$TASK/$TASK-$hard-stage-adapter-prompt-bias.out

    # ADAPTER BIAS PROMPT
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 3e-3 1e-3 3e-4
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG$hard \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            TRAIN_PARAM='adapter bias prompt' \
            bash run_mulstage_experiment.sh "--use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG$hard', 'task_name': '$task', 'training_params': ['adapter', 'prompt', 'bias']}" > ./result/$TASK/$TASK-$hard-stage-adapter-bias-prompt.out
done

if [[ $task == mnli ]]; then
    task=mnli-mm
    for hard in Y N 
    do
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'do_train': False}" > ./result/$TASK/$TASK-mm-$hard-none.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt'}" > ./result/$TASK/$TASK-mm-$hard-prompt.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'bias'}" > ./result/$TASK/$TASK-mm-$hard-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'adapter'}" > ./result/$TASK/$TASK-mm-$hard-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt,adapter'}" > ./result/$TASK/$TASK-mm-$hard-prompt-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt,bias'}" > ./result/$TASK/$TASK-mm-$hard-prompt-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'bias,adapter'}" > ./result/$TASK/$TASK-mm-$hard-bias-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': 'prompt,bias,adapter'}" > ./result/$TASK/$TASK-mm-$hard-prompt-bias-adapter.out
    done
fi
