# Major Function of Model Module 
- [Major Function of Model Module](#major-function-of-model-module)
  - [BaseModel](#basemodel)
  - [from_pretrain](#from_pretrain)
    - [load model from modelhub](#load-model-from-modelhub)
    - [load local model weights](#load-local-model-weights)
  - [All supported models](#all-supported-models)
  - [Supported models + tasks](#supported-models--tasks)
  - [Model design](#model-design)
  - [forward function](#forward-function)
    - [input](#input)
    - [output](#output)
  - [init_from_json](#init_from_json)
    - [input](#input-1)
    - [output](#output-1)

## BaseModel

The base class BaseModel implements common methods for loading/saving models from a local file or directory or from a library-provided pretrained model configuration (downloaded from BAAI modelhub's Kingsoft S3 repository).
All supported models now support the three most common model types [encoder, decoder and encoder-decoder]. GLM models can now load all GLM series models, see https://github.com/THUDM/GLM

## from_pretrain

`from_pretrain` is used for load models. Models with the same model structure can be loaded with the same class. For example, `BERT-base` and `Roberta-base` models can be loaded with the `BertModel` class. `from_pretrain` is optimized for data/model parallel model loading to avoid resource waste caused by repeated downloads.

### load model from modelhub

By calling `ClassName.from_pretrian()` to load following [supported models](#all-supported-models), it will automatically download the model configuration file `config.json`, model weights `pytorch_model.bin`, and dictionary files `vocab .txt`.

```python
# Downloading GLM-large-ch from modelhub
from flagai.model.glm_model import GLMModel
model = GLMModel.from_pretrain(download_path="./state_dict", model_name="GLM-large-ch")
```
### load local model weights
If the model weights are stored in `download_path/model_name/`, where `model_name` is the directory where the model is located, and `download_path` is the directory where `model_name` is located,

```python
# load  `pytorch_model.bin` and `config.json` from `./state_dict/GLM-large-ch`
from flagai.model.glm_model import GLMModel
model = GLMModel.from_pretrain(download_path="./state_dict", model_name="GLM-large-ch")
```

## All supported models
We can use `from_pretrain` to automatically download following models


| ClassName                                 | ModelName | Language | Model Type  |
|-------------------------------------------|-----------|----------|-------------|
| flagai.model.glm_model.GLMModel | ** GLM-10b-ch** | chinese | encoder |
| flagai.model.glm_model.GLMModel | **GLM-large-ch** | chinese | encoder |
| flagai.model.bert_model.BertModel | **RoBERTa-base-ch** | chinese | encoder |
| flagai.model.gpt2_model.GPT2Model | **GPT2-base-ch** | chinese | decoder |
| flagai.model.t5_model.T5Model | **T5-base-ch** | chinese | enc2dec |
| flagai.model.t5_model.T5Model | **T5-base-en** | chinese | enc2dec |
| flagai.model.bert_model.BertModel | **BERT-base-en** | english | encoder |
| flagai.model.glm_model.GLMModel | **GLM-large-en** | english | encoder |

## Supported models + tasks

At the same time, we support the finetuned model on the task, as shown in the table below, the model weights can be loaded through `ClassName.from_pretrain()`, for example, we automatically download and load a `GLM-large-ch` model finetuned on the title-generation task:

````python
from flagai.model.glm_model import GLMForSeq2Seq
model = GLMForSeq2Seq.from_pretrain(model_name='GLM-large-ch')
````

We also provide the `AutoLoader` class to help load models. For example, the `GLM-large-ch` model is used for `seq2seq` tasks. Here we adopt a task- and model-independent design. In theory, tasks and models can be freely replaced.

````python
from flagai.auto_model.auto_loader import AutoLoader
auto_loader = AutoLoader("title-generation",
                         model_name="GLM-large-ch",
                         model_dir= "./state_dict")
model = auto_loader.get_model()
````

| ClassName | Model Name | language | Task |
|------------------------------------------------- |-----------------|----------|-------------------|
| flagai.model.glm_model.GLMForSeq2Seq | GLM-large-ch | chinese | **title generation** |
| flagai.model.glm_model.GLMForSeq2Seq | GLM-large-ch | chinese | **poetry generation** |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese | **title generation** |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese | **NER** |
| flagai.model.bert_model.BertForSequenceLabeling | RoBERTa-base-ch | chinese | **semantic matching** |
| flagai.model.t5_model.T5Model | T5-base-ch | chinese | **title generation** |
| flagai.model.bert_model.BertForSequenceLabeling | BERT-base-en | english | **title gneration** |

## Model design

The main construction logic of the model `layer->block->model`

`flagai.model.layer`: including mlp, layernorm, activation, attention and other layers

`flagai.model.block`: Build a transformer block by assembling various layers, such as BERT block, etc.

`flagai.model`: build the model by embedding layers and stacked blocks

If you want to customize a new model structure, you can refer to the above construction process.

## forward function

### input
Input parameters are keyword arguments: including input_ids, position_ids, attention_mask, etc., redundant parameters will be automatically ignored
For example, GLM's forward function:

```python
def forward(self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            mems=None,
            return_memory=False,
            detach_memory=True,
            prompt_pos=None,
            **kwargs)
```

### output
The output is a dictionary, including logits and hidden states, which are required, such as the return of the GLM forword function:

```python
return {'loss': loss, 'logits': logits, 'hidden_states': mems}
```

## init_from_json

### input
The input is a json and `**kwargs` which is used for the new initialization parameters of models.
For example, the initialization of GLMModel is as follows:

```python
GLMModel.init_from_json(config_file = "./config.json", checkpoint_activations=True)
```
`checkpoint_activations=True` is a new parameter used to control whether to perform gradient recomputation.

### output
An instance of GLM model.