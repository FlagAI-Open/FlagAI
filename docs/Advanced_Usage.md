## Advanced usage

### custom model
If you want to customize the model or improve the model in the FlagAI framework, for example, follow the following rules:

1. Inheriting BaseModel, BaseModel supports loading parameters in the form of from_pretrain and building models with different parameters in the form of init_from_json.
2. The first parameter of the ``__init__()`` function of the custom model must be config, and each key is the parameter information in the ``config.json`` file.
3. There must be a ``load_weights()`` function in the custom model. The input is the address of the pre-training parameters, and the pre-training parameters are loaded for the custom model.
4. The ``forward()`` function in the custom model must return a dictionary, which must contain ``logits``. If the labels parameter is passed in, it needs to additionally return ``loss`` data.

Take GLM to complete the sequence labeling task as an example:

```python
from flagai.model.base_model import BaseModel
from flagai.model.glm_model import GLMModel
import torch

class GLMForSequenceClassification(BaseModel):
    def __init__(self, config, hidden_dropout=0.1, pool_token='cls', **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.pool_token = pool_token
        self.model = GLMModel(config)
        self.model.output_predict = False
        self.num_class = config['class_num']
        # Multi-choice head.
        hidden_size = self.model.hidden_size
        self.pool_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.multichoice_dropout = torch.nn.Dropout(hidden_dropout)
        self.multichoice_head = torch.nn.Linear(hidden_size, self.num_class)

    def forward(self,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                **kwargs):
        num_choices = None
        if len(input_ids.shape) == 3:
            assert self.num_class == 1
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1,
                                                    *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
        model_out = self.model(input_ids, position_ids, attention_mask)
        outputs, mems = model_out['logits'], model_out['hidden_states']
        if self.pool_token == 'start':
            output = outputs[torch.arange(outputs.size(0),
                                          dtype=attention_mask.dtype,
                                          device=attention_mask.device),
                             attention_mask]
        elif self.pool_token == 'pad':
            output = outputs[torch.arange(outputs.size(0),
                                          dtype=attention_mask.dtype,
                                          device=attention_mask.device),
                             attention_mask - 1]
        elif self.pool_token == 'cls':
            output = outputs[:, 0]
        else:
            raise NotImplementedError
        output = torch.tanh(self.pool_layer(output))
        multichoice_output = self.multichoice_dropout(output)
        logits = self.multichoice_head(multichoice_output)
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        if 'labels' not in kwargs:
            return {'logits': logits, 'hidden_states': mems}
        else:
            labels = kwargs['labels']
            if logits.size(1) == 1:
                logits = logits.squeeze(1)
                loss = F.binary_cross_entropy_with_logits(
                    logits.contiguous().float(), labels.float())
            else:
                loss = F.cross_entropy(logits.contiguous().float(),
                                       labels.long())
            return {"loss": loss, 'logits': logits, 'hidden_states': mems}

    def compute_loss(self,
                     input_ids=None,
                     position_ids=None,
                     attention_mask=None,
                     labels=None,
                     **kwargs):
        model_output = self.forward(input_ids=input_ids,
                                    position_ids=position_ids,
                                    attention_mask=attention_mask)
        assert labels is not None, "labels must not None!"
        logits = model_output['logits']
        loss = F.cross_entropy(logits.contiguous().float(), labels.long())
        return {
            "loss": loss,
            'logits': model_output['logits'],
            'hidden_states': model_output['hidden_states']
        }

    def load_weights(self, checkpoint_path):
        checkpoints = self.model.load_weights_glm(checkpoint_path)
        self.load_state_dict(checkpoints, strict=False)
```

Among them, in the ``__init__()`` function, except for the config parameter, the other parameters are flexibly defined for different tasks.
For example, ``hidden_dropout`` and ``pool_token`` in the following code, these parameters can be flexibly passed in when calling ``from_pretrain()``, for example:
```python
from flagai.model.glm_model import GLMForSequenceClassification
model_dir = "./state_dict/GLM_sequence_classification/" ## this dir is the position for model and vocab and config files.
model = GLMForSequenceClassification.from_pretrain(model_dir, 
                                                   hidden_dropout=0.1,
                                                   pool_token="cls")
```

After building a custom model and ensuring that the pre-training parameters are loaded correctly, you can use it directly in the framework.

### Choose the right Tokenizer

After customizing the model, you need to select a suitable Tokenizer to cooperate with the model training. FlagAI supports a variety of native Tokenizers, such as:

1. BertTokenizer: ```from flagai.data.tokenizer.bert_tokenizer import BertTokenizer``` BertTokenizer supports a variety of models, including Chinese models, English models; Bert models, RoBERTa models, Chinese GPT2 models, and more.
2. GLMLargeChTokenizer: ```from flagai.data.tokenizer.glm_large_ch_tokenizer import GLMLargeChTokenizer``` GLMLargeChTokenizer supports ``GLM-large-ch`` model。
3. GLMLargeEnTokenizer: ```from flagai.data.tokenizer.glm_large_en_tokenizer import GLMLargeEnTokenizer``` GLMLargeEnTokenizer supports ``GLM-large-en`` model。
3. T5BPETokenizer: ```from flagai.data.tokenizer.t5_tokenizer import T5BPETokenizer``` T5BPETokenizer supports englisht T5 model：``T5-base-en``。
4. T5PegasusTokenizer: ```from flagai.data.tokenizer.t5_pegasus_tokenizer import T5PegasusTokenizer``` T5PegasusTokenizer supports chinese T5 model：``T5-base-ch``。