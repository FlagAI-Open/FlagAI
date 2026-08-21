"""Auto RoBERTa model"""

import torch
import torch.nn as nn
from packaging import version

from transformers import RobertaForMaskedLM, PreTrainedModel
from .model import PromptRobertaForMaskedLM


class AutoRobertaForMaskedLM(nn.Module):
    def __init__(self, config, use_prompt, model_name_or_path=None):
        super().__init__()

        self.use_prompt = use_prompt
        if model_name_or_path is not None:
            if use_prompt :
                self.roberta = PromptRobertaForMaskedLM.from_pretrained(model_name_or_path, config=config)
            else:
                self.roberta = RobertaForMaskedLM.from_pretrained(model_name_or_path, config=config)
        else:
            if use_prompt:
                self.roberta = PromptRobertaForMaskedLM(config=config)
            else:
                self.roberta = RobertaForMaskedLM(config=config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        self.num_labels = config.num_labels
        self.evaluate = True

    def freeze_model(self):
        self.evaluate = True
        for p in self.parameters():
            p.requires_grad = False

    def train_prompt(self):
        if not hasattr(self, 'training_params') or self.training_params != "prompt":
            self.evaluate = True
            self.params = []
            for n, p in self.named_parameters():
                if 'prompt' in n:
                    self.params.append(p)

        self.training_params = "prompt"
        if self.evaluate is True:
            self.evaluate = False
            for p in self.params:
                p.requires_grad = True

    def add_adapter(self, *args, **kwargs):
        return self.roberta.add_adapter(*args, **kwargs)

    def train_adapter(self, *args, **kwargs):
        self.training_params = "adapter"
        if self.evaluate is True:
            self.evaluate = False
            self.adapter_args = args
            self.adapter_kwargs = kwargs
            return self.roberta.train_adapter(*args, **kwargs)

    def train_bias(self):
        if not hasattr(self, 'training_params') or self.training_params != "bias":
            self.evaluate = True
            self.params = []
            for n, p in self.named_parameters():
                if 'bias' in n:
                    self.params.append(p)

        self.training_params = "bias"
        if self.evaluate is True:
            self.evaluate = False
            for p in self.params:
                p.requires_grad = True

    def train(self, mode=True):
        if mode == False:
            self.evaluate = True
            super().train(mode)
            return
        if not hasattr(self, 'training_params'):
            super().train(mode)
            return
        if self.training_params == "adapter":
            self.train_adapter(*self.adapter_args, **self.adapter_kwargs)
        elif self.training_params == "bias":
            self.train_bias()
        elif self.training_params == "prompt":
            self.train_prompt()
        else:
            super().train(mode)

    def eval(self):
        self.evaluate = True
        super().eval()

    def forward(self, input_ids=None, attention_mask=None, mask_pos=None, labels=None):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        output = self.roberta(input_ids, attention_mask=attention_mask)
        prediction_scores = output[1]
        prediction_mask_scores = prediction_scores[torch.arange(prediction_scores.size(0)), mask_pos]

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


