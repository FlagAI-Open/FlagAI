from openpromptu.data_utils import InputExample
import torch
from transformers.data.data_collator import torch_default_data_collator
from transformers.data.data_collator import DataCollatorMixin as HfDataCollatorMixin
from transformers.data.data_collator import DataCollatorForSeq2Seq as DataCollator
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
import copy
from torch.nn import CrossEntropyLoss

def preprocess_function(raw_example, **kwargs):
    tokenizer = kwargs['tokenizer']
    data_args = kwargs['data_args']
    template = kwargs['template']
    verbalizer = kwargs['verbalizer']
    tokenizer_wrapper = kwargs['tokenizer_wrapper']

    example = InputExample(**raw_example)
    # example = verbalizer.wrap_one_example(example)
    example, other = template.wrap_one_example(example)
    input_sentence = tokenizer_wrapper.merge_wrapped_example(example)
    model_inputs = tokenizer(input_sentence, max_length=data_args.max_source_length,
                        padding="max_length", truncation=True)
    return model_inputs
    


def compute_metrics(eval_preds, dataset_name, eval_metric):
    pass

def mask_token_func(tokenizer, ith_mask=0):
    return tokenizer.pad_token

def get_remove_columns(dataset_features):
    # dataset_features.remove("label")
    return dataset_features

def get_prompts(task, tokenizer, data_args, template_id="0", verbalizer_id="0"):
    from openpromptu.prompts import GenerationVerbalizer
    from openpromptu.prompts import ManualTemplate
    from openpromptu import TokenizerWrapper
    template = ManualTemplate(text = task.templates_text[template_id])
    verbalizer = GenerationVerbalizer(tokenizer=tokenizer, classes = None, label_words=None)
    tokenizer_wrapper = TokenizerWrapper(max_seq_length=data_args.max_source_length, tokenizer=tokenizer, truncate_method="tail", mask_token_func=mask_token_func)
    return template, verbalizer, tokenizer_wrapper


def get_backbone(model_args, **kwargs):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if not hasattr(tokenizer,"pad_token") or (hasattr(tokenizer,"pad_token") and tokenizer.pad_token==None):
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        )

    return config, tokenizer, model

class Trainer(HfSeq2SeqTrainer):
    def __init__(self, verbalizer=None, eval_task=None, **kwargs):
        super().__init__(**kwargs)
        self.eval_task = eval_task
        self.compute_metrics = self._compute_metrics

    def compute_loss(self, model, inputs, return_outputs=False):

        labels=copy.deepcopy(inputs['input_ids'])
        # labels[labels==self.tokenizer.pad_token_id]=-100
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.long().view(-1))

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model, #nn.Module,
        inputs, #Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only, #: bool,
        ignore_keys, #: Optional[List[str]] = None,
    ): #-> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            labels=copy.deepcopy(inputs['input_ids'])
            # labels[labels==self.tokenizer.pad_token_id]=-100
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().long()
            loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)).detach().cpu()
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
            
        if prediction_loss_only:
            return (loss, None, None)
        else:
            # non pad label
            shift_labels = shift_labels.view(-1).detach().cpu()
            nonpad_idx = shift_labels!=self.tokenizer.pad_token_id
            shift_labels = shift_labels[nonpad_idx]
            # the probability at the corresponding position
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])[nonpad_idx].detach().cpu()
            target_position = torch.nn.functional.one_hot(shift_labels,shift_logits.shape[-1]).bool().to(shift_labels.device)
            shift_logits = shift_logits.softmax(dim=-1)[target_position]


            return (loss, shift_logits, shift_labels)

    def _compute_metrics(self, eval_preds):

        preds, labels = eval_preds

        result = {}
        for metric in self.eval_task.metric:
            result.update(metric(preds, labels,ignore_index=self.tokenizer.pad_token_id))

        average_metric = sum(result.values())/len(result)
        result.update({"average_metrics":average_metric})
        return result