# Required environment variables:
# TASK: SST-2 / CoLA / MNLI / QNLI / RTE / MRPC / QQP / STS-B


for TASK in CoLA SST-2 MNLI STS-B MRPC QQP QNLI RTE
do
    model=roberta-large

    [ ! -d "./result/$TASK-fulldata-gap" ] && mkdir ./result/$TASK-fulldata-gap

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
        TAG=exp-fulldata-16-$hard

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'do_train': False}" > ./result/$TASK-fulldata-gap/$TASK-$hard-none.out

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt']}" > ./result/$TASK-fulldata-gap/$TASK-$hard-prompt.out

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias']}" > ./result/$TASK-fulldata-gap/$TASK-$hard-bias.out

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['adapter']}" > ./result/$TASK-fulldata-gap/$TASK-$hard-adapter.out

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,adapter']}" > ./result/$TASK-fulldata-gap/$TASK-$hard-prompt-adapter.out

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias']}" > ./result/$TASK-fulldata-gap/$TASK-$hard-prompt-bias.out

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['bias,adapter']}" > ./result/$TASK-fulldata-gap/$TASK-$hard-bias-adapter.out

        python tools/gather_result_gen_gap.py --condition "{'tag': '$TAG', 'task_name': '$task', 'training_params': ['prompt,bias,adapter']}" > ./result/$TASK-fulldata-gap/$TASK-$hard-prompt-bias-adapter.out
    done

done