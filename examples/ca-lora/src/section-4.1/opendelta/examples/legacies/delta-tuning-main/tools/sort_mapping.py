import argparse
import json
import numpy as np
import torch
from torch import device
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)")
    parser.add_argument('--mapping_dir', type=str, help='Mapping directory')

    # These options should be kept as their default values
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--log", type=str, default="log", help="Log path.")
    parser.add_argument("--key", type=str, default='', help="Validation metric name")
    parser.add_argument("--test_key", type=str, default="", help="Test metric name")
    parser.add_argument("--test_key2", type=str, default="", help="Second test metric name")

    args = parser.parse_args()

    condition = eval(args.condition)

    if len(args.key) == 0:
        if condition['task_name'] == 'cola':
            args.key = 'cola_dev_eval_mcc'
            args.test_key = 'cola_test_eval_mcc'
            print_name = 'CoLA'
        elif condition['task_name'] == 'mrpc/acc':
            args.key = 'mrpc_dev_eval_acc'
            args.test_key = 'mrpc_test_eval_acc'
            args.test_key2 = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
            print_name = 'MRPC'
        elif condition['task_name'] == 'mrpc/f1':
            args.key = 'mrpc_dev_eval_f1'
            args.test_key2 = 'mrpc_test_eval_acc'
            args.test_key = 'mrpc_test_eval_f1'
            condition['task_name'] = 'mrpc'
            print_name = 'MRPC'
        elif condition['task_name'] == 'qqp/acc':
            args.key = 'qqp_dev_eval_acc'
            args.test_key = 'qqp_test_eval_acc'
            args.test_key2 = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
            print_name = 'QQP'
        elif condition['task_name'] == 'qqp/f1':
            args.key = 'qqp_dev_eval_f1'
            args.test_key2 = 'qqp_test_eval_acc'
            args.test_key = 'qqp_test_eval_f1'
            condition['task_name'] = 'qqp'
            print_name = 'QQP'
        elif condition['task_name'] == 'sts-b/pearson':
            args.key = 'sts-b_dev_eval_pearson'
            args.test_key = 'sts-b_test_eval_pearson'
            args.test_key2 = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
            print_name = 'STS-B'
        elif condition['task_name'] == 'sts-b/spearmanr':
            args.key = 'sts-b_dev_eval_spearmanr'
            args.test_key2 = 'sts-b_test_eval_pearson'
            args.test_key = 'sts-b_test_eval_spearmanr'
            condition['task_name'] = 'sts-b'
            print_name = 'STS-B'
        elif condition['task_name'] == 'qnli':
            args.key = 'qnli_dev_eval_acc'
            args.test_key = 'qnli_test_eval_acc'
            print_name = 'QNLI'
        elif condition['task_name'] == 'sst-2':
            args.key = 'sst-2_dev_eval_acc'
            args.test_key = 'sst-2_test_eval_acc'
            print_name = 'SST-2'
        elif condition['task_name'] == 'snli':
            args.key = 'snli_dev_eval_acc'
            args.test_key = 'snli_test_eval_acc'
            print_name = 'SNLI'
        elif condition['task_name'] == 'mnli':
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli_test_eval_mnli/acc'
            print_name = 'MNLI'
        elif condition['task_name'] == 'mnli-mm':
            condition['task_name'] = 'mnli'
            args.key = 'mnli_dev_eval_mnli/acc'
            args.test_key = 'mnli-mm_test_eval_mnli-mm/acc'
            print_name = 'MNLI'
        elif condition['task_name'] == 'rte':
            args.key = 'rte_dev_eval_acc'
            args.test_key = 'rte_test_eval_acc'
            print_name = 'RTE'
        elif condition['task_name'] == 'ag_news':
            args.key = 'ag_news_dev_eval_acc'
            args.test_key = 'ag_news_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'yahoo_answers':
            args.key = 'yahoo_answers_dev_eval_acc'
            args.test_key = 'yahoo_answers_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'yelp_review_full':
            args.key = 'yelp_review_full_dev_eval_acc'
            args.test_key = 'yelp_review_full_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'mr':
            args.key = 'mr_dev_eval_acc'
            args.test_key = 'mr_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'sst-5':
            args.key = 'sst-5_dev_eval_acc'
            args.test_key = 'sst-5_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'subj':
            args.key = 'subj_dev_eval_acc'
            args.test_key = 'subj_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'trec':
            args.key = 'trec_dev_eval_acc'
            args.test_key = 'trec_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'cr':
            args.key = 'cr_dev_eval_acc'
            args.test_key = 'cr_test_eval_acc'
            print_name = condition['task_name']
        elif condition['task_name'] == 'mpqa':
            args.key = 'mpqa_dev_eval_acc'
            args.test_key = 'mpqa_test_eval_acc'
            print_name = condition['task_name']
        else:
            raise NotImplementedError

    with open(args.log) as f:
        result_list = []
        for line in f:
            result_list.append(eval(line))
    
    seed_result = {}
    seed_result_mapping_id = {} # avoid duplication

    for item in result_list:
        ok = True
        for cond in condition:
            if cond not in item or item[cond] != condition[cond]:
                ok = False
                break
        
        if ok:
            seed = item['seed']
            if seed not in seed_result:
                seed_result[seed] = [item]
                seed_result_mapping_id[seed] = {item['mapping_id']: 1}
            else:
                if item['mapping_id'] not in seed_result_mapping_id[seed]:
                    seed_result[seed].append(item)
                    seed_result_mapping_id[seed][item['mapping_id']] = 1

    for seed in seed_result:
        print("Seed %d has %d results" % (seed, len(seed_result[seed])))

        # Load all mappings
        with open(os.path.join(args.mapping_dir, print_name, "{}-{}.txt".format(args.k, seed))) as f:
            mappings = []
            for line in f:
                mappings.append(line.strip())

        # Write sorted mappings
        fsort = open(os.path.join(args.mapping_dir, print_name, "{}-{}.sort.txt".format(args.k, seed)), 'w')
        fscore = open(os.path.join(args.mapping_dir, print_name, "{}-{}.score.txt".format(args.k, seed)), 'w')

        seed_result[seed].sort(key=lambda x: x[args.key], reverse=True)
        for item in seed_result[seed]:
            fsort.write(mappings[item['mapping_id']] + '\n')
            fscore.write("%.5f %s\n" % (item[args.key], mappings[item['mapping_id']]))

if __name__ == '__main__':
    main()
