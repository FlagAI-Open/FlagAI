import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import string
from collections import Counter
import bmtrain as bmt
from rouge import Rouge

def softmax_acc(score, label, acc_result):
    if acc_result is None or acc_result["total"] > 25600:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())
    return acc_result

def mlm_acc_loss(predict, labels, acc_result, loss):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0, "loss": []}
    acc_result["right"] += int((predict[labels > 0] == labels[labels > 0]).sum())
    acc_result["total"] += int((labels > 0).sum())
    # if loss == 0 and len(acc_result["loss"]) > 0:
    #     loss = torch.tensor(sum(acc_result["loss"]) / len(acc_result["loss"]), device=predict.device)
    if loss != 0:
        acc_result["loss"].append(bmt.sum_loss(loss).item())
        acc_result["loss"] = acc_result["loss"][-100:]
    return acc_result

def microf1(scores, labels, acc_result):
    if acc_result is None:
        acc_result = {"TP": 0, "FP": 0, "FN": 0}
    # scores: batch, label_num
    # labels: batch, label_num
    predict = scores > 0.5
    acc_result["TP"] += int(labels[predict].sum())
    acc_result["FP"] += int((labels[predict] == 0).sum())
    acc_result["FN"] += int(labels[scores <= 0.5].sum())
    return acc_result



def normalize_answer(s):
    # return s
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def ROUGE_normalize_answer(s):
    # return s
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def squad_em(predict, answers):
    em = 0
    for pre, ans in zip(predict, answers):
        if pre in ans:
            em += 1
        # else:
        #     print("predict: %s\t answer: %s" % (pre, ans))
    return em

def squad_f1(predict, answers):
    ret = 0
    for pred, ans in zip(predict, answers):
        # if pred == "no answer":
        #     continue
        prediction_tokens = pred.split()
        cpred_token = Counter(prediction_tokens)
        curf1 = []
        for a in ans:
            ground_truth_tokens = a.split()
            common = cpred_token & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                curf1.append(0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                curf1.append(f1)
        ret += max(curf1)
    return ret

def squad_NAF1(predict, answers, acc_result):
    for p, ans in zip(predict, answers):
        if p == "no answer":
            if "no answer" in ans:
                acc_result["NA_tp"] += 1
            else:
                acc_result["NA_fp"] += 1
        else:
            if "no answer" in ans:
                acc_result["NA_fn"] += 1
            else:
                acc_result["NA_tn"] += 1
    return acc_result

def squad_metric(predict, answers, acc_result, RL=False):
    if acc_result is None:
        acc_result = {"train": False, "total": 0, "em_sum": 0, "f1_sum": 0., "NA_tp": 0, "NA_fp": 0, "NA_tn": 0, "NA_fn": 0, "ROUGE-L-R": 0, "ROUGE-L-P": 0, "ROUGE-L-F": 0}
    pred = []
    rouge_pred = []
    for s in predict:
        if RL:
            rouge_pred.append(ROUGE_normalize_answer(s))
        pred.append(normalize_answer(s))
    # pred = [normalize_answer([int(n) for n in p if n == 1 break], skip_special_tokens=True)) for p in predict]
    ground = [{normalize_answer(a) for a in ans} for ans in answers]

    if RL:
        ROUGE_ground = [ROUGE_normalize_answer(list(ans)[0]) for ans in answers]
        scorer = Rouge()
        score = scorer.get_scores([p if p != "" else " " for p in rouge_pred], ROUGE_ground, avg=True)
        acc_result["ROUGE-L-P"] += score["rouge-l"]["p"] * len(rouge_pred)
        acc_result["ROUGE-L-R"] += score["rouge-l"]["r"] * len(rouge_pred)
        acc_result["ROUGE-L-F"] += score["rouge-l"]["f"] * len(rouge_pred)

    acc_result["em_sum"] += squad_em(pred, ground)
    acc_result["f1_sum"] += squad_f1(pred, ground)
    acc_result["total"] += len(pred)
    acc_result = squad_NAF1(pred, ground, acc_result)
    # print(acc_result)
    return acc_result

def squad_train_metric(predict, labels, acc_result):
    # predict: batch, len
    # labels: batch, len
    if acc_result is None:
        acc_result = {"train": True, "total": 0, "right": 0}
    acc_result["right"] += int((predict[labels > 0] == labels[labels > 0]).sum())
    acc_result["total"] += int((labels > 0).sum())
    return acc_result

def sum_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()
    ret = white_space_fix(lower(s))
    if ret == "":
        ret = " "
    return ret

def summarization_metric(predict, answers, acc_result, tokenizer):
    if acc_result is None:
        acc_result = {"train": False, "total": 0, "rouge-1": 0.0, "rouge-2": 0.0, "rouge-3": 0.0}
    pred = []
    for p in predict:
        tmp = []
        for n in p:
            if n == 1:
                break
            tmp.append(int(n))
        pred.append(sum_normalize_answer(tokenizer.decode(tmp, skip_special_tokens=True)))
    ground = [sum_normalize_answer(ans) for ans in answers]
    # print(pred)
    scorer = Rouge()
    score = scorer.get_scores(pred, ground)
    acc_result["rouge-1"] = score[0]["rouge-1"]["r"] * len(pred)
    acc_result["rouge-2"] = score[0]["rouge-2"]["r"] * len(pred)
    acc_result["rouge-l"] = score[0]["rouge-l"]["r"] * len(pred)
    acc_result["total"] += len(pred)
    # print(score)
    # print(acc_result)
    return acc_result
