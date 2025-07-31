from openpromptu.data_utils import InputExample
import torch
from transformers.data.data_collator import torch_default_data_collator
from transformers.data.data_collator import DataCollatorMixin as HfDataCollatorMixin
import numpy as np
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
)

from transformers import Trainer as HfTrainer
import torch.nn as nn


def get_prompts(task, tokenizer, data_args, template_id="0", verbalizer_id="0"):
    # from openpromptu.prompts import ManualVerbalizer
    # from openpromptu.prompts import ManualTemplate
    # from openpromptu import TokenizerWrapper
    # template = ManualTemplate(text = task.templates_text[template_id])
    # verbalizer = ManualVerbalizer(tokenizer=tokenizer, classes = task.labels_list, label_words=task.verbalizers[verbalizer_id])
    # tokenizer_wrapper = TokenizerWrapper(max_seq_length=data_args.max_source_length, tokenizer=tokenizer, truncate_method="balanced", mask_token_func=mask_token_func)
    return None, None, None

def preprocess_function(raw_example, **kwargs):
    # from IPython import embed; embed(header="Therefa")
    tokenizer = kwargs['tokenizer']
    # print(np.array(raw_example['img']).shape)
    model_inputs = tokenizer(np.array(raw_example['image']), return_tensors='pt')
    model_inputs['pixel_values'] = model_inputs['pixel_values'].squeeze()
    model_inputs['labels'] = raw_example['label']
    return model_inputs

def compute_metrics(eval_preds, dataset_name, eval_metric):
    # from IPython import embed; embed(header="In compute metrics")

    preds, labels = eval_preds.predictions, eval_preds.label_ids

    preds = np.argmax(preds, axis=-1)

    result = {}
    average_metrics = []
    for metric in eval_metric:
        metric_item = metric(preds, labels)
        metric_value =  list(metric_item.values())
        result.update(metric_item)
        average_metrics.extend(metric_value)
    print("average:",average_metrics)
    average_metric = sum(average_metrics)/len(average_metrics)
    result.update({"average_metrics":average_metric})
    return result

def mask_token_func(tokenizer, ith_mask=0):
    return tokenizer.mask_token

def get_remove_columns(dataset_features):
    # dataset_features.pop("label")
    # print("remove_columns: {}".format(dataset_features))
    return dataset_features

class DataCollator(HfDataCollatorMixin):
    def __init__(self, *args, **kwargs):
        self.return_tensors='pt'

    def torch_call(self, features):
        # from IPython import embed; embed(header="in data collator")
        a = torch_default_data_collator(features=features)
        # from IPython import embed; embed(header="in data collator")
        return a


def get_backbone(model_args, **kwargs):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.dropout_rate = 0.0
    tokenizer = AutoFeatureExtractor.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,

        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.num_labels = model_args.num_classes
    old_classifier = model.classifier
    model.classifier = nn.Linear(old_classifier.in_features, config.num_labels)


    return config, tokenizer, model

class Trainer(HfTrainer):
    def __init__(self, verbalizer=None, eval_task=None, **kwargs):
        super().__init__(**kwargs)
        self.verbalizer=verbalizer
        self.eval_task=eval_task
        self.compute_metrics = self._compute_metrics
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _compute_metrics(self, eval_preds):
        # from IPython import embed; embed(header="In compute metrics")

        preds, labels = eval_preds.predictions, eval_preds.label_ids

        preds = np.argmax(preds, axis=-1)

        result = {}
        average_metrics = []
        for metric in self.eval_task.metric:
            metric_item = metric(preds, labels)
            metric_value =  list(metric_item.values())
            result.update(metric_item)
            average_metrics.extend(metric_value)
        print("average:",average_metrics)
        average_metric = sum(average_metrics)/len(average_metrics)
        result.update({"average_metrics":average_metric})
        from IPython import embed; embed(header="In compute metrics")
        return result


