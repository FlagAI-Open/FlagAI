import argparse
import os
import numpy as np
import pandas as pd
from torch import device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)")
    
    # These options should be kept as their default values
    parser.add_argument("--log", type=str, default="log", help="Log path.")
    parser.add_argument("--key", type=str, default='', help="Validation metric name")
    parser.add_argument("--test_key", type=str, default="", help="Test metric name")
    parser.add_argument("--test_key2", type=str, default="", help="Second test metric name")

    args = parser.parse_args()

    condition = eval(args.condition)

    print(condition)
    tag = condition['tag'][-1]
    index = ['cola', 'sst-2', 'mrpc/f1', 'sts-b/pearson', 'qqp', 'mnli', 'qnli', 'rte', 'average']
    column = None
    if 'training_params' not in condition:
        column = 0
    else:
        column = 0
        if 'prompt' in condition['training_params']:
            column += 4
        if 'bias' in condition['training_params']:
            column += 2
        if 'adapter' in condition['training_params']:
            column += 1
        if column == 0:
            raise ValueError

    if len(args.key) == 0:
        if condition['task_name'] == 'cola':
            args.train_key = 'cola_train_eval_mcc'
            args.key = 'cola_dev_eval_mcc'
            args.test_key = 'cola_test_eval_mcc'
        elif condition['task_name'] == 'mrpc/acc':
            args.train_key = 'mrpc_train_eval_acc'
            args.key = 'mrpc_dev_eval_acc'
            args.test_key = 'mrpc_test_eval_acc'
            args.test_key2 = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'mrpc/f1':
            args.train_key = 'mrpc_train_eval_f1'
            args.key = 'mrpc_dev_eval_f1'
            args.test_key2 = 'mrpc_test_eval_acc'
            args.test_key = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'qqp/acc':
            args.train_key = 'qqp_train_eval_acc'
            args.key = 'qqp_dev_eval_acc'
            args.test_key = 'qqp_test_eval_acc'
            args.test_key2 = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'qqp/f1':
            args.train_key = 'qqp_train_eval_f1'
            args.key = 'qqp_dev_eval_f1'
            args.test_key2 = 'qqp_test_eval_acc'
            args.test_key = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'sts-b/pearson':
            args.train_key = 'sts-b_train_eval_pearson'
            args.key = 'sts-b_dev_eval_pearson'
            args.test_key = 'sts-b_test_eval_pearson'
            args.test_key2 = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'sts-b/spearmanr':
            args.train_key = 'sts-b_train_eval_spearmanr'
            args.key = 'sts-b_dev_eval_spearmanr'
            args.test_key2 = 'sts-b_test_eval_pearson'
            args.test_key = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'qnli':
            args.train_key = 'qnli_train_eval_acc'
            args.key = 'qnli_dev_eval_acc'
            args.test_key = 'qnli_test_eval_acc'
        elif condition['task_name'] == 'sst-2':
            args.train_key = 'sst-2_train_eval_acc'
            args.key = 'sst-2_dev_eval_acc'
            args.test_key = 'sst-2_test_eval_acc'
        elif condition['task_name'] == 'mnli':
            args.train_key = 'mnli_train_eval_mnli/acc'
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli_test_eval_mnli/acc'
        elif condition['task_name'] == 'mnli-mm':
            args.train_key = 'mnli_train_eval_mnli/acc'
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli-mm_test_eval_mnli-mm/acc'
        elif condition['task_name'] == 'rte':
            args.train_key = 'rte_train_eval_acc'
            args.key = 'rte_dev_eval_acc'
            args.test_key = 'rte_test_eval_acc'
        else:
            raise NotImplementedError

    with open(args.log) as f:
        result_list = []
        for line in f:
            result_list.append(eval(line))
    
    seed_result = {}
    seed_best = {}
    lrs = {}

    total = 0
    for item in result_list:
        ok = True
        for cond in condition:
            if cond not in item or (item[cond] != condition[cond]):
                ok = False
                break

        if ok:
            total = total+1
            seed = item['data_dir'].split('-')[-1] + '-' + str(item['seed'])
            if item['learning_rate'] not in lrs:
                lrs[item['learning_rate']] = {seed: item}
            else:
                if seed not in lrs[item['learning_rate']]:
                    lrs[item['learning_rate']][seed] = item
                else:
                    print('Warning: Overwirte results on learning rate = '+
                           str(item['learning_rate']) +', seed = ' + str(seed, dtype=np.float64) )
                    lrs[item['learning_rate']][seed] = item

            # if seed not in seed_result:
            #     seed_result[seed] = [item]
            #     seed_best[seed] = item
            # else:
            #     seed_result[seed].append(item)
            #     if item[args.key] > seed_best[seed][args.key]:
            #         seed_best[seed] = item

    print('total experiments: %d\n' % total)

    answer=None
    min_lr = 1
    max_lr = 0
    best_lr = None
    std = None
    gpmean = None
    gpstd = None

    # s = []
    # for lr in lrs:
    #     for seed in lrs[lr]:
    #         item = lrs[lr][seed]
    #         if args.test_key in item:
    #             s.append(item[args.test_key])
    #         else:
    #             print('Use Validation result!')
    #             s.append(item[args.key])
    # print(s)
    # if len(s) == 1:
    #     print({'mean': s[0], 'std': 0})
    # else:
    #     s.remove(min(s))
    #     print(s)
    #     print({'mean': np.array(s).mean(), 'std': np.array(s).std()})

    for lr in lrs:
        s = []
        gap = []
        for seed in lrs[lr]:
            item = lrs[lr][seed]
            if args.test_key in item:
                s.append(item[args.test_key])
                gap.append(abs(item[args.test_key]-item[args.train_key]))
                print({"seed": seed, "learning_rate": lr, "train_result": item[args.train_key], "test_result": item[args.test_key]})
            else:
                print('Use Validation result!')
                s.append(item[args.key])
                gap.append(abs(item[args.key]-item[args.train_key]))
                print({"seed": seed, "learning_rate": lr, "train_result": item[args.train_key], "dev_result": item[args.key]})
        test_acc_mean = np.array(s).mean()
        test_acc_std = np.array(s).std()
        gap_mean = np.array(gap).mean()
        gap_std = np.array(gap).std()
        print(f'Statistics for learning rate = {lr}: mean: {test_acc_mean}, std: {test_acc_std}, gap_mean: {gap_mean}, gap_std: {gap_std}')
        if answer==None or test_acc_mean > answer:
            answer = test_acc_mean
            best_lr = lr
            std = test_acc_std
            gpmean = gap_mean
            gpstd = gap_std
        min_lr = min(min_lr, lr)
        max_lr = max(max_lr, lr)
    if best_lr == min_lr:
        print('Warning! Best Learning rate equals minmum Learning rate.')
    elif best_lr == max_lr:
        print('Warning! Best Learning rate equals maximum Learning rate.')
    print({'best_lr': best_lr, 'mean': answer, 'std': std, 'gap_mean': gpmean, 'gap_std': gpstd})

    row = index.index(condition['task_name'])
    print(column)
    # Store in csv
    table = os.path.join('./', f'gap-{tag}.csv')
    if not os.path.exists(table):
        d = [[0 for i in range(8)] for j in range(9)]
        frame = pd.DataFrame(data=d, dtype=np.float64)
    else:
        frame = pd.read_csv(table, header=None, dtype=np.float64)
    frame.at[row, column] = gpmean
    frame.to_csv(table, index=False, header=False)

    table = os.path.join('./', f'gap_std-{tag}.csv')
    if not os.path.exists(table):
        d = [[0 for i in range(8)] for j in range(9)]
        frame = pd.DataFrame(data=d, dtype=np.float64)
    else:
        frame = pd.read_csv(table, header=None, dtype=np.float64)
    frame.at[row, column] = gpstd
    frame.to_csv(table, index=False, header=False)

    table = os.path.join('./', f'perf-{tag}.csv')
    if not os.path.exists(table):
        d = [[0 for i in range(8)] for j in range(9)]
        frame = pd.DataFrame(data=d, dtype=np.float64)
    else:
        frame = pd.read_csv(table, header=None, dtype=np.float64)
    frame.at[row, column] = answer
    frame.to_csv(table, index=False, header=False)

    table = os.path.join('./', f'perf_std-{tag}.csv')
    if not os.path.exists(table):
        d = [[0 for i in range(8)] for j in range(9)]
        frame = pd.DataFrame(data=d, dtype=np.float64)
    else:
        frame = pd.read_csv(table, header=None, dtype=np.float64)
    frame.at[row, column] = std
    frame.to_csv(table, index=False, header=False)


if __name__ == '__main__':
    main()
