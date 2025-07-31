import argparse
import json
import numpy as np
import torch
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

    if len(args.key) == 0:
        if condition['task_name'] == 'cola':
            args.key = 'cola_dev_eval_mcc'
            args.test_key = 'cola_test_eval_mcc'
        elif condition['task_name'] == 'mrpc/acc':
            args.key = 'mrpc_dev_eval_acc'
            args.test_key = 'mrpc_test_eval_acc'
            args.test_key2 = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'mrpc/f1':
            args.key = 'mrpc_dev_eval_f1'
            args.test_key2 = 'mrpc_test_eval_acc'
            args.test_key = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
        elif condition['task_name'] == 'qqp/acc':
            args.key = 'qqp_dev_eval_acc'
            args.test_key = 'qqp_test_eval_acc'
            args.test_key2 = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'qqp/f1':
            args.key = 'qqp_dev_eval_f1'
            args.test_key2 = 'qqp_test_eval_acc'
            args.test_key = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
        elif condition['task_name'] == 'sts-b/pearson':
            args.key = 'sts-b_dev_eval_pearson'
            args.test_key = 'sts-b_test_eval_pearson'
            args.test_key2 = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'sts-b/spearmanr':
            args.key = 'sts-b_dev_eval_spearmanr'
            args.test_key2 = 'sts-b_test_eval_pearson'
            args.test_key = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
        elif condition['task_name'] == 'qnli':
            args.key = 'qnli_dev_eval_acc'
            args.test_key = 'qnli_test_eval_acc'
        elif condition['task_name'] == 'sst-2':
            args.key = 'sst-2_dev_eval_acc'
            args.test_key = 'sst-2_test_eval_acc'
        elif condition['task_name'] == 'snli':
            args.key = 'snli_dev_eval_acc'
            args.test_key = 'snli_test_eval_acc'
        elif condition['task_name'] == 'mnli':
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli_test_eval_mnli/acc'
        elif condition['task_name'] == 'mnli-mm':
            condition['task_name'] = 'mnli'
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli-mm_test_eval_mnli-mm/acc'
        elif condition['task_name'] == 'rte':
            args.key = 'rte_dev_eval_acc'
            args.test_key = 'rte_test_eval_acc'
        elif condition['task_name'] == 'ag_news':
            args.key = 'ag_news_dev_eval_acc'
            args.test_key = 'ag_news_test_eval_acc'
        elif condition['task_name'] == 'yahoo_answers':
            args.key = 'yahoo_answers_dev_eval_acc'
            args.test_key = 'yahoo_answers_test_eval_acc'
        elif condition['task_name'] == 'yelp_review_full':
            args.key = 'yelp_review_full_dev_eval_acc'
            args.test_key = 'yelp_review_full_test_eval_acc'
        elif condition['task_name'] == 'mr':
            args.key = 'mr_dev_eval_acc'
            args.test_key = 'mr_test_eval_acc'
        elif condition['task_name'] == 'sst-5':
            args.key = 'sst-5_dev_eval_acc'
            args.test_key = 'sst-5_test_eval_acc'
        elif condition['task_name'] == 'subj':
            args.key = 'subj_dev_eval_acc'
            args.test_key = 'subj_test_eval_acc'
        elif condition['task_name'] == 'trec':
            args.key = 'trec_dev_eval_acc'
            args.test_key = 'trec_test_eval_acc'
        elif condition['task_name'] == 'cr':
            args.key = 'cr_dev_eval_acc'
            args.test_key = 'cr_test_eval_acc'
        elif condition['task_name'] == 'mpqa':
            args.key = 'mpqa_dev_eval_acc'
            args.test_key = 'mpqa_test_eval_acc'
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
                           str(item['learning_rate']) +', seed = ' + str(seed) )
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
        for seed in lrs[lr]:
            item = lrs[lr][seed]
            if args.test_key in item:
                s.append(item[args.test_key])
            else:
                print('Use Validation result!')
                s.append(item[args.key])
            print({"seed": seed, "learning_rate": lr, "dev_result": item[args.key], "test_result": item[args.key]})
        test_acc_mean = np.array(s).mean()
        test_acc_std = np.array(s).std()
        print('Statistics for learning rate = '+str(lr)+": mean: "+str(test_acc_mean)+", std: " + str(test_acc_std))
        if answer==None or test_acc_mean > answer:
            answer = test_acc_mean
            best_lr = lr
            std = test_acc_std
        min_lr = min(min_lr, lr)
        max_lr = max(max_lr, lr)
    if best_lr == min_lr:
        print('Warning! Best Learning rate equals minmum Learning rate.')
    elif best_lr == max_lr:
        print('Warning! Best Learning rate equals maximum Learning rate.')
    print({'best_lr': best_lr, 'mean': answer, 'std': std})

    # for seed in seed_result:
    #     seed_list = seed_result[seed]
    #     for item in seed_list:
            # print("seed = %s, learning_rate = %f, results = %.4f" % (seed, item['learning_rate'], item[args.test_key]))
    
    # final_result_dev = np.zeros((len(seed_best)))
    # final_result_test = np.zeros((len(seed_best)))
    # final_result_test2 = np.zeros((len(seed_best)))
    # for i, seed in enumerate(seed_best):
    #     final_result_dev[i] = seed_best[seed][args.key]
    #     final_result_test[i] = seed_best[seed][args.test_key]
    #     if len(args.test_key2) > 0:
    #         final_result_test2[i] = seed_best[seed][args.test_key2]
    #     print("%s: best dev (%.5f) test (%.5f) %s | total trials: %d" % (
    #         seed,
    #         seed_best[seed][args.key],
    #         seed_best[seed][args.test_key],
    #         "test2 (%.5f)" % (seed_best[seed][args.test_key2]) if len(args.test_key2) > 0 else "",
    #         len(seed_result[seed])
    #     ))
    #     s = ''
    #     for k in ['per_device_train_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'eval_steps', 'max_steps']:
    #         s += '| {}: {} '.format(k, seed_best[seed][k])
    #     print('    ' + s)

    # s = "mean +- std: %.1f (%.1f) (median %.1f)" % (final_result_test.mean() * 100, final_result_test.std() * 100, np.median(final_result_test) * 100)
    # if len(args.test_key2) > 0:
    #     s += "second metric: %.1f (%.1f) (median %.1f)" % (final_result_test2.mean() * 100, final_result_test2.std() * 100, np.median(final_result_test2) * 100)
    # print(s)

if __name__ == '__main__':
    main()
