import abc
from collections import OrderedDict
import numpy as np

"""Defines functions to process the outputs to make them ready for the evaluation."""

def string_to_float(string, default=-1., **unused_kwargs):
  """Converts string to float, using default when conversion not possible."""
  try:
    return float(string)
  except ValueError:
    return default


class PostProcessor(abc.ABC): 
    """Postprocess the predictions and labels to make them suitable for
    evaluation."""
    def __init__(self, tokenizer, ignore_pad_token_for_loss):
       self.tokenizer = tokenizer 
       self.ignore_pad_token_for_loss = ignore_pad_token_for_loss 

    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels 


class MultiRC(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info) 
        preds = [{"group": info["group"], "value":pred} \
            for info, pred in zip(data_info, preds)]
        labels = [{"group": info["group"], "value": label}\
            for info, label in zip(data_info, labels)] 
        return preds, labels 

class Record(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info) 
        labels = [info["answers"] for info in data_info]
        return preds, labels 


POSTPROCESSOR_MAPPING = OrderedDict(
    [
        ('superglue-record', Record),
        ('superglue-multirc', MultiRC)
    ]
)

class AutoPostProcessor:
    @classmethod
    def get(self, task, tokenizer, ignore_pad_token_for_loss):
        if task in POSTPROCESSOR_MAPPING:
            return POSTPROCESSOR_MAPPING[task](tokenizer, ignore_pad_token_for_loss)
        return PostProcessor(tokenizer, ignore_pad_token_for_loss)
