import argparse
import pandas as pd
import json
import numpy as np
import torch
import os
from torch import device
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments, glue_compute_metrics
from transformers.data.metrics import simple_accuracy
from transformers.data.processors.glue import glue_processors

def get_glue_label(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[1]
        elif task == 'MNLI':
            return line[-1]
        elif task == 'MRPC':
            return line[0]
        elif task == 'QNLI':
            return line[-1]
        elif task == 'QQP':
            return line[-1]
        elif task == 'RTE':
            return line[-1]
        elif task == 'SNLI':
            return line[-1]
        elif task == 'SST-2':
            return line[-1]
        elif task == 'STS-B':
            return 0 if float(line[-1]) < 2.5 else 1
        elif task == 'WNLI':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def get_labels(data_dir, k, seed, task, print_name):
    if print_name in ['sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec']:
        data = pd.read_csv(os.path.join(data_dir, print_name, '{}-{}'.format(k, seed), 'test.csv'), header=None).values.tolist()
        labels = np.zeros((len(data)), dtype=np.uint8)
        for i, example in enumerate(data):
            labels[i] = example[0]
    elif print_name in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        lines = []
        file_name = os.path.join(data_dir, print_name, '{}-{}'.format(k, seed), 'test.tsv')
        if task == 'mnli':
            file_name = os.path.join(data_dir, print_name, '{}-{}'.format(k, seed), 'test_matched.tsv')
        elif task == 'mnli-mm':
            file_name = os.path.join(data_dir, print_name, '{}-{}'.format(k, seed), 'test_mismatched.tsv')

        for line in open(file_name):
            lines.append(line.strip())

        if task != 'cola':
            lines = lines[1:]
        label_list = glue_processors[task]().get_labels()
        label_map = {k: i for i, k in enumerate(label_list)}
        if task == 'sts-b':
            label_map = {0: 0, 1: 1}
        label_ids = np.zeros((len(lines)))
        for line_id, line in enumerate(lines):
            label_ids[line_id] = label_map[get_glue_label(print_name, line)]
    return label_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_models", type=int, help="Number of models")
    parser.add_argument("--k", type=int, default=16, help="Number of training instances per label")
    parser.add_argument("--condition", type=str, help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)")
    
    # These options should usually be kept as their default values
    parser.add_argument("--data_dir", type=str, default="data/k-shot", help="Data directory")
    parser.add_argument("--save_logit_dir", type=str, default="ensemble_predict_results", help="Directory to store the logit file.")
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
    
    # Gather all logs satisfying the conditions
    for item in result_list:
        ok = True
        for cond in condition:
            if cond == 'task_name' and condition['task_name'] == 'mnli-mm':
                if cond not in item or item[cond] != 'mnli':
                   ok = False
                   break
            else: 
                if cond not in item or item[cond] != condition[cond]:
                    ok = False
                    break
        if 'model_id' not in item or 'array_id' not in item:
            ok = False
        
        if ok:
            seed = int(item['data_dir'].split('-')[-1])
            model_id = item['model_id']
            array_id = item['array_id']
           
            if model_id >= 0 and model_id < args.n_models:
                if seed not in seed_result:
                    seed_result[seed] = {}
                    seed_best[seed] = {}
                if model_id not in seed_result[seed]:
                    seed_result[seed][model_id] = []
                    seed_best[seed][model_id] = {args.key: -1e9} 

                seed_result[seed][model_id].append(item)
                if item[args.key] > seed_best[seed][model_id][args.key]:
                    seed_best[seed][model_id] = item
    
    final_result_dev = np.zeros((len(seed_result), args.n_models))
    final_result_test = np.zeros((len(seed_result), args.n_models))
    final_result_test2 = np.zeros((len(seed_result), args.n_models))

    logit_file_list = {}
    for seed in seed_result:
        logit_file_list[seed] = []

    # Get the results for each model and pick the best dev trial for each model/seed
    for model_id in range(args.n_models):
        for i, seed in enumerate(seed_result):
            final_result_dev[i][model_id] = seed_best[seed][model_id][args.key]
            final_result_test[i][model_id] = seed_best[seed][model_id][args.test_key]
            if len(args.test_key2) > 0:
                final_result_test2[i][model_id] = seed_best[seed][model_id][args.test_key2]

            logit_file_list[seed].append("{}-{}-{}.npy".format(condition['task_name'], model_id, seed_best[seed][model_id]["array_id"]))

        s = "Model %d | val: mean +- std: %.1f +- %.1f | test: mean +- std: %.1f (%.1f) (median %.1f)" % (model_id, final_result_dev[:, model_id].mean() * 100, final_result_dev[:, model_id].std() * 100, final_result_test[:, model_id].mean() * 100, final_result_test[:, model_id].std() * 100, np.median(final_result_test[:, model_id]) * 100)
        if len(args.test_key2) > 0:
            s += " / %.1f +- %.1f (median %.1f)" % (final_result_test2[:, model_id].mean() * 100, final_result_test2[:, model_id].std() * 100, np.median(final_result_test2[:, model_id]) * 100)
        print(s)

    # Map lower-case names to official names (data folder name)
    data_dir_mapping = {
        'cola': 'CoLA',
        'mrpc': 'MRPC',
        'qqp': 'QQP',
        'sts-b': 'STS-B',
        'sst-2': 'SST-2',
        'snli': 'SNLI',
        'mnli': 'MNLI',
        'mnli-mm': 'MNLI',
        'rte': 'RTE',
        'ag_news': 'ag_news',
        'yahoo_answers': 'yahoo_answers',
        'yelp_review_full': 'yelp_review_full',
        'sst-5': 'sst-5',
        'mr': 'mr',
        'cr': 'cr',
        'mpqa': 'mpqa',
        'subj': 'subj',
        'trec': 'trec'
    }

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    ensemble_result = np.zeros((len(seed_result)))
    ensemble_result2 = np.zeros((len(seed_result))) # for second metric

    # Ensemble for each seed
    for seed_id, seed in enumerate(seed_result):
        labels = get_labels(args.data_dir, args.k, seed, condition['task_name'], data_dir_mapping[condition['task_name']])

        # Logits
        mean_logits = None
        for fname in logit_file_list[seed]:
            logits = np.load(os.path.join(args.save_logit_dir, fname))
            if mean_logits is None:
                mean_logits = logits
            else:
                mean_logits += logits
        mean_logits /= len(logit_file_list[seed])
        
        # Compute metrics
        preds = mean_logits.argmax(-1)
        if condition['task_name'] in ['sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec']:
            metric = {"acc": simple_accuracy(preds, labels)}
        else:
            metric = glue_compute_metrics(condition['task_name'], preds, labels)

        ensemble_result[seed_id] = metric[args.test_key.split('_')[-1]]
        if len(args.test_key2) > 0:
            ensemble_result2[seed_id] = metric[args.test_key2.split('_')[-1]]
     
    s = "mean +- std: %.1f (%.1f) (median %.1f)" % (ensemble_result.mean() * 100, ensemble_result.std() * 100, np.median(ensemble_result) * 100)
    if len(args.test_key2) > 0:
        s += " / %.1f (%.1f) (median %.1f)" % (ensemble_result2.mean() * 100, ensemble_result2.std() * 100, np.median(ensemble_result2) * 100)
    print(s)

if __name__ == '__main__':
    main()
