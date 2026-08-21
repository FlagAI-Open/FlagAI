from openpromptu.data_utils import InputExample
import torch
from transformers.data.data_collator import torch_default_data_collator
from transformers.data.data_collator import DataCollatorMixin as HfDataCollatorMixin
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from transformers import Trainer as HfTrainer


def preprocess_function(raw_example, **kwargs):
    tokenizer = kwargs['tokenizer']
    data_args = kwargs['data_args']
    template = kwargs['template']
    verbalizer = kwargs['verbalizer']
    tokenizer_wrapper = kwargs['tokenizer_wrapper']

    example = InputExample(**raw_example)
    example, other = template.wrap_one_example(example)
    input_sentence = tokenizer_wrapper.merge_wrapped_example(example)
    model_inputs = tokenizer(input_sentence, max_length=data_args.max_source_length,
                        padding="max_length", truncation=True)
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
    dataset_features.remove("label")
    return dataset_features


def get_prompts(task, tokenizer, data_args, template_id="0", verbalizer_id="0"):
    from openpromptu.prompts import ManualVerbalizer
    from openpromptu.prompts import ManualTemplate
    from openpromptu import TokenizerWrapper
    template = ManualTemplate(text = task.templates_text[template_id])
    verbalizer = ManualVerbalizer(tokenizer=tokenizer, classes = task.labels_list, label_words=task.verbalizers[verbalizer_id])
    tokenizer_wrapper = TokenizerWrapper(max_seq_length=data_args.max_source_length, tokenizer=tokenizer, truncate_method="balanced", mask_token_func=mask_token_func)
    # from IPython import embed; embed()
    return template, verbalizer, tokenizer_wrapper

class DataCollator(HfDataCollatorMixin):
    def __init__(self, *args, **kwargs):
        self.return_tensors='pt'

    def torch_call(self, features):
        return torch_default_data_collator(features=features)



def get_backbone(model_args, **kwargs):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))
    return config, tokenizer, model

class Trainer(HfTrainer):
    def __init__(self, verbalizer=None, eval_task=None, **kwargs):
        super().__init__(**kwargs)
        self.verbalizer=verbalizer
        self.eval_task=eval_task
        self.compute_metrics = self._compute_metrics


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get("logits")
        input_ids = inputs['input_ids']
        verbalizer = self.verbalizer.cuda()
        logits_at_mask = logits[torch.where(input_ids == verbalizer.tokenizer.mask_token_id)]
        label_logits = verbalizer.process_logits(logits_at_mask)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(label_logits, labels)
        outputs.logits = label_logits
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
        return result


