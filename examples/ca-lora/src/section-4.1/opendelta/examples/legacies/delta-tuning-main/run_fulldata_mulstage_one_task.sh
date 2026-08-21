# Required environment variables:
# TASK: SST-2 / CoLA / MNLI / QNLI / RTE / MRPC / QQP / STS-B

model=roberta-large
cuda=0,1,4,7
bs=64
gpun=4

[ ! -d "./result/$TASK-fulldata-mulstage" ] && mkdir ./result/$TASK-fulldata-mulstage

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
    TAG=exp-fulldata-mulstage-$hard

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
            bash run_fulldata_mulstage_experiment.sh "--training_params prompt bias adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt', 'bias', 'adapter']}" > ./result/$TASK-fulldata-mulstage/$TASK-$hard-prompt-bias-adapter.out

    # Prompt + Adapter + Bias
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
            bash run_fulldata_mulstage_experiment.sh "--training_params prompt adapter bias --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt', 'adapter', 'bias']}" > ./result/$TASK-fulldata-mulstage/$TASK-$hard-prompt-adapter-bias.out

    # Adapter + Bias + Prompt
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
            bash run_fulldata_mulstage_experiment.sh "--training_params adapter bias prompt --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter', 'bias', 'prompt']}" > ./result/$TASK-fulldata-mulstage/$TASK-$hard-adapter-bias-prompt.out

    # Adapter + Prompt + Bias
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
            bash run_fulldata_mulstage_experiment.sh "--training_params adapter prompt bias --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter', 'prompt', 'bias']}" > ./result/$TASK-fulldata-mulstage/$TASK-$hard-adapter-prompt-bias.out

    # Bias + Prompt + Adapater
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
            bash run_fulldata_mulstage_experiment.sh "--training_params bias prompt adapter --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias', 'prompt', 'adapter']}" > ./result/$TASK-fulldata-mulstage/$TASK-$hard-adapter-bias-prompt.out

    # Bias + Adapater + Prompt
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
            bash run_fulldata_mulstage_experiment.sh "--training_params bias adapter prompt --use_adapter"
            if (($? != 0)); then exit 0; fi
        done
    done

    python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias', 'adapter', 'prompt']}" > ./result/$TASK-fulldata-mulstage/$TASK-$hard-adapter-bias-prompt.out
done

if [[ $task == mnli ]]; then
    task=mnli-mm
    for hard in Y N 
    do
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'do_train': False}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-none.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt']}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-prompt.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias']}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter']}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,adapter']}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-prompt-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias']}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-prompt-bias.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias,adapter']}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-bias-adapter.out
        python tools/gather_result.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias,adapter']}" > ./result/$TASK-fulldata-mulstage/$TASK-mm-$hard-prompt-bias-adapter.out
    done
fi
