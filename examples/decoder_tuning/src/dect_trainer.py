import os, shutil
import sys
sys.path.append(".")

import torch
from torch import nn
from tqdm import tqdm
import dill
import warnings

from typing import Callable, Union, Dict
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict
from sklearn.metrics import accuracy_score
from openprompt.pipeline_base import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger

class DecTRunner(object):
    r"""A runner for DecT
    This class is specially implemented for classification.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptForClassification`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
    """
    def __init__(self, 
                 model: PromptForClassification,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None,
                 calibrate_dataloader: Optional[PromptDataLoader] = None,
                 id2label: Optional[Dict] = None,
                 verbalizer = None,
                 ):
        self.model = model.cuda()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.calibrate_dataloader = calibrate_dataloader
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.id2label = id2label
        self.verbalizer = verbalizer
        self.clean = True
    
    def inference_step(self, batch, batch_idx):
        label = batch.pop('label')
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()
    
    def inference_epoch(self, split: str): 
        outputs = []
        scores = {}
        self.model.eval()
        with torch.no_grad():
            data_loader = self.valid_dataloader if split=='validation' else self.test_dataloader
            model_preds, preds, labels = self.verbalizer.test(self.model, data_loader)
            # zs_score = accuracy_score(labels, model_preds)
            score = accuracy_score(labels, preds)
            scores = {"dect acc": score}
        return scores

    def inference_epoch_end(self, outputs):
        preds = []
        labels = []
        for pred, label in outputs:
            preds.extend(pred)
            labels.extend(label)

        score = accuracy_score(labels, preds)
        return score

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.loss_function(logits, batch['label'])
        return loss

    def fit(self, ckpt: Optional[str] = None):

        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")

        self.model.verbalizer.train_proto(self.model, self.train_dataloader, self.calibrate_dataloader)

        return 0
    
    def test(self, ckpt: Optional[str] = None) -> dict:
        if ckpt:
            if not self.load_checkpoint(ckpt, load_state = False):
                exit()
        return self.inference_epoch("test")

    def run(self, ckpt: Optional[str] = None) -> dict:
        self.fit(ckpt)
        return self.test(ckpt = None if self.clean else 'best')