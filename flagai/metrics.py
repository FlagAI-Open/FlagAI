# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import re
from sklearn.metrics import f1_score
from collections import defaultdict
from typing import List
import functools
import string
import math


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


def accuracy_metric(predictions, labels, meta=None):
    count = 0
    assert len(predictions) == len(labels)
    if predictions.size() != labels.size():
        predictions = torch.argmax(predictions, dim=-1)
        for prediction, label in zip(predictions, labels):
            count += prediction == label
    else:
        for prediction, label in zip(predictions, labels):
            if sigmoid(prediction) >= 0.5:
                count += label == 1
            else:
                count += label == 0
    return 100.0*count/len(labels)


def f1_metric(predictions, labels, meta=None):
    pred = torch.argmax(predictions, dim=-1).cpu()
    labels = labels.cpu()
    if torch.equal(pred, labels):
        return 1.0
    return f1_score(labels, pred)


def f1_macro_metric(predictions, labels, meta=None):
    pred = torch.argmax(predictions, dim=-1).cpu()
    labels = labels.cpu()
    if torch.equal(pred, labels):
        return 1.0
    return f1_score(labels, pred, average='macro')


def multirc_em(predictions, labels, meta):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    question_ids = meta["question_idx"]
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, labels))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)
    return em


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return 0.0
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def qa_evaluate(predictions, labels, meta, metric):
    # assert len(examples) == len(predictions)
    score = 0.0
    for ground_truths, candidate, prediction in zip(meta["answers"],
                                                    meta['candidates'],
                                                    predictions):
        # ground_truths = example.meta["answers"]
        prediction = candidate[prediction]
        if ground_truths:
            score += metric_max_over_ground_truths(metric, prediction,
                                                   ground_truths)
    score = 100.0 * score / len(predictions)
    return score


qa_exact_match = functools.partial(qa_evaluate, metric=exact_match_score)
qa_f1 = functools.partial(qa_evaluate, metric=f1_score)
