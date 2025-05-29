
from openpromptu.data_utils import InputExample
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import (
    AutoConfig,
    BlenderbotForConditionalGeneration,
    AutoTokenizer,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq as DataCollator
import torch

def mask_token_func(tokenizer, ith_mask=0):
    return ""

def get_remove_columns(dataset_features):
    return dataset_features

def preprocess_function(raw_example, **kwargs):
    # max_target_length += 1
    tokenizer = kwargs['tokenizer']
    data_args = kwargs['data_args']
    template = kwargs['template']
    verbalizer = kwargs['verbalizer']
    tokenizer_wrapper = kwargs['tokenizer_wrapper']
    split = kwargs['split']
    example = InputExample(**raw_example)


   
    example = verbalizer.wrap_one_example(example)
    example, other = template.wrap_one_example(example)
    input_sentence = tokenizer_wrapper.merge_wrapped_example(example)
    model_inputs = tokenizer(input_sentence, max_length=data_args.max_source_length,
                        padding="max_length", truncation=True)


    with tokenizer.as_target_tokenizer():
        label = tokenizer(other['tgt_text']).input_ids

    model_inputs["labels"] = label
    # from IPython import embed; embed()
    return model_inputs

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


    model = BlenderbotForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        )
    # from IPython import embed; embed()
    return config, tokenizer, model


def get_prompts(task, tokenizer, data_args, template_id="blenderbot", verbalizer_id="blenderbot"):
    from openpromptu.prompts import GenerationVerbalizer
    from openpromptu.prompts import ManualTemplate
    from openpromptu import TokenizerWrapper
    template = ManualTemplate(text = task.templates_text[template_id])
    verbalizer = GenerationVerbalizer(tokenizer=tokenizer, classes = task.labels_list, label_words=task.verbalizers[verbalizer_id])
    tokenizer_wrapper = TokenizerWrapper(max_seq_length=data_args.max_source_length, tokenizer=tokenizer, truncate_method="balanced", mask_token_func=mask_token_func)
    return template, verbalizer, tokenizer_wrapper

class Trainer(HfSeq2SeqTrainer):
    def __init__(self, verbalizer=None, eval_task=None, **kwargs):
        super().__init__(**kwargs)
        self.eval_task = eval_task
        self.compute_metrics = self._compute_metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        # from IPython import embed; embed()
        outputs = model(**inputs)
        if return_outputs:
            return (outputs.loss, outputs)
        else:
            return outputs.loss

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


        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": 10, # self._max_length if s is not None else self.model.config.max_length,
            "num_beams": 1, #self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "min_length": 1  # for blenderbot, generally we set it to be a large number. But in classification, we set it to 1
        }
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():

            outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        # from IPython import embed; embed(header="In seqseqtrainer")
        return (loss, generated_tokens, labels)

    def _compute_metrics(self, eval_preds):
        # from IPython import embed; embed(header="In compute metrics")
        preds, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # post_processor = .get(data_args.dataset_name[0], tokenizer,
        #                                     data_args.ignore_pad_token_for_loss)
        # decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        result = {}
        for metric in self.eval_task.metric:
            result.update(metric(decoded_preds, decoded_labels))

        average_metric = sum(result.values())/len(result)
        result.update({"average_metrics":average_metric})
        return result

