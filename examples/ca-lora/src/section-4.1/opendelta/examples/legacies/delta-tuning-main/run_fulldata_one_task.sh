# Required environment variables:
# TASK: SST-2 / CoLA / MNLI / QNLI / RTE / MRPC / QQP / STS-B

model=roberta-large
cuda=1,2,4,5
bs=64
gpun=4

[ ! -d "./result/$TASK-fulldata" ] && mkdir ./result/$TASK-fulldata

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
    TAG=exp-fulldata-$hard

    # None
    CUDA_VISIBLE_DEVICES=$cuda \
    TASK=$TASK \
    TAG=$TAG \
    BS=$bs \
    SEED=13 \
    MODEL=$model \
    HARD=$hard \
    NOTRAIN=1 \
    GPUN=$gpun \
    bash run_fulldata_experiment.sh

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'do_train': False}" > ./result/$TASK-fulldata/$TASK-$hard-none.out

    # PROMPT
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            bash run_fulldata_experiment.sh "--training_params prompt"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt']}" > ./result/$TASK-fulldata/$TASK-$hard-prompt.out

    # Bias
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            bash run_fulldata_experiment.sh "--training_params bias"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias']}" > ./result/$TASK-fulldata/$TASK-$hard-bias.out

    # Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            bash run_fulldata_experiment.sh "--training_params adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter']}" > ./result/$TASK-fulldata/$TASK-$hard-adapter.out

    # Prompt + Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            bash run_fulldata_experiment.sh "--training_params prompt,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,adapter']}" > ./result/$TASK-fulldata/$TASK-$hard-prompt-adapter.out

    # Prompt + Bias
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            bash run_fulldata_experiment.sh "--training_params prompt,bias"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias']}" > ./result/$TASK-fulldata/$TASK-$hard-prompt-bias.out

    # Bias + Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            bash run_fulldata_experiment.sh "--training_params bias,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias,adapter']}" > ./result/$TASK-fulldata/$TASK-$hard-bias-adapter.out

    # Prompt + Bias + Adapter
    for seed in 13 21 42 87 100
    do
        for lr in 1e-2 1e-3 1e-4 1e-5
        do
            CUDA_VISIBLE_DEVICES=$cuda \
            TASK=$TASK \
            TAG=$TAG \
            BS=$bs \
            LR=$lr \
            PROMPT=prompt \
            SEED=$seed \
            MODEL=$model \
            HARD=$hard \
            GPUN=$gpun \
            bash run_fulldata_experiment.sh "--training_params prompt,bias,adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias,adapter']}" > ./result/$TASK-fulldata/$TASK-$hard-prompt-bias-adapter.out
done

if [[ $task == mnli ]]; then
    task=mnli-mm
    for hard in Y N 
    do
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'do_train': False}" > ./result/$TASK-fulldata/$TASK-mm-$hard-none.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt']}" > ./result/$TASK-fulldata/$TASK-mm-$hard-prompt.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias']}" > ./result/$TASK-fulldata/$TASK-mm-$hard-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter']}" > ./result/$TASK-fulldata/$TASK-mm-$hard-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,adapter']}" > ./result/$TASK-fulldata/$TASK-mm-$hard-prompt-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias']}" > ./result/$TASK-fulldata/$TASK-mm-$hard-prompt-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias,adapter']}" > ./result/$TASK-fulldata/$TASK-mm-$hard-bias-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias,adapter']}" > ./result/$TASK-fulldata/$TASK-mm-$hard-prompt-bias-adapter.out
    done
fi
