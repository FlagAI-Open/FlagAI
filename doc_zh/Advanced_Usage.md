## 高级用法

### 自定义模型
如果想在FlagAI框架中自定义模型或改进模型，比如遵循以下几条规则：

1. 继承BaseModel，BaseModel 支持以from_pretrain的方式加载参数，以init_from_json的方式构建不同参数的模型。
2. 自定义模型的```__init__()```函数第一个参数必须为config，其中每一个key为``config.json``文件中的参数信息。
3. 自定义模型中必须有``load_weights()``函数，输入为预训练参数地址，为自定义模型加载预训练参数。
4. 自定义模型中``forward()``函数必须返回字典，其中必须包含``logits``，如果传入labels参数，则需要额外返回``loss``数据。

以GLM完成序列标注任务为例：

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

其中``__init__()``函数中除了config参数必填，其他参数针对不同任务灵活定义，例如如下代码中的 ``hidden_dropout`` 与 ``pool_token``，这些参数在使用``from_pretrain()``函数构建模型的时候，可以灵活进行传入，例如：

```python
from flagai.model.glm_model import GLMForSequenceClassification
model_dir = "./state_dict/GLM_sequence_classification/" ## this dir is the position for model and vocab and config files.
model = GLMForSequenceClassification.from_pretrain(model_dir, 
                                                   hidden_dropout=0.1,
                                                   pool_token="cls")
```

构建好自定义模型，保证预训练参数加载正确，便可以直接在框架中进行使用。


### 选择合适的 Tokenizer

自定义好模型之后，需要选择一个合适的Tokenizer配合模型训练，FlagAI中支持多种原生Tokenizer，例如：

1. BertTokenizer: ```from flagai.data.tokenizer.bert_tokenizer import BertTokenizer``` BertTokenizer支持多种模型，包括中文模型，英文模型；Bert模型，RoBERTa模型，中文GPT2模型等等。
2. GLMLargeChTokenizer: ```from flagai.data.tokenizer.glm_large_ch_tokenizer import GLMLargeChTokenizer``` GLMLargeChTokenizer 支持 ``GLM-large-ch`` 模型。
3. GLMLargeEnTokenizer: ```from flagai.data.tokenizer.glm_large_en_tokenizer import GLMLargeEnTokenizer``` GLMLargeEnTokenizer 支持 ``GLM-large-en`` 模型。
3. T5BPETokenizer: ```from flagai.data.tokenizer.t5_tokenizer import T5BPETokenizer``` T5BPETokenizer 支持英文T5模型：``T5-base-en``。
4. T5PegasusTokenizer: ```from flagai.data.tokenizer.t5_pegasus_tokenizer import T5PegasusTokenizer``` T5PegasusTokenizer 支持中文T5模型：``T5-base-ch``。