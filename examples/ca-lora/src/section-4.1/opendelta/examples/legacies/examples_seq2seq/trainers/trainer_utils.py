import numpy as np 
from typing import Union, NamedTuple, Tuple, Dict, Any   
import os 
import regex as re
import logging
from dataclasses import fields
import torch.nn as nn
import json




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
        data_info: (:obj:`Dict[str, Any]`): Extra dataset information, one requires
        to performs the evaluation. The data_info is a dictionary with keys from
        train, eval, test to specify the data_info for each split of the dataset.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    data_info: Dict[str, Any]





def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_last_checkpoint(output_dir):
    if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
        return output_dir
    return None


def pad_punctuation(text):
   """Re-implementation of _pad_punctuation in t5. This function adds spaces
   around punctuation. While this pads punctuation as expected, it has the 
   unexpected effected of padding certain unicode characters with accents, with
   spaces as well. For instance: "François" becomes "Fran ç ois"""
   # Pad everything except for: underscores (_), whitespace (\s),
   # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
   text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', text)
   # Collapse consecutive whitespace into one space.
   text = re.sub(r'\s+', ' ', text)
   return text

def save_json(filepath, dictionary):
   with open(filepath, "w") as outfile:
      json.dump(dictionary, outfile)


def read_json(filepath):
   f = open(filepath,)
   return json.load(f)


def save_training_config(config_file, output_dir):
   json_data = read_json(config_file)
   save_json(os.path.join(output_dir, "training_config.json"), json_data)

