(autodelta)=
# AutoDelta Mechanism

Inspired by [Huggingface transformers AutoClasses](https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/auto#transformers.AutoModel) , we provide an AutoDelta features for the users to

1. Easily to experiment with different delta models
2. Fast deploy from configuration file, especially from the repos in [DeltaCenter](https://examplelink).


## Easily load from dict, so that subject to change the type of delta models.

```python
from opendelta import AutoDeltaConfig, AutoDeltaModel
from transformers import T5ForConditionalGeneration

backbone_model = T5ForConditionalGeneration.from_pretrained("t5-base")
```

We can load a config from a dict
```python
config_dict = {
    "delta_type":"lora", 
    "modified_modules":[
        "SelfAttention.q", 
        "SelfAttention.v",
        "SelfAttention.o"
    ], 
    "lora_r":4}
delta_config = AutoDeltaConfig.from_dict(config_dict)
```

Then use the config to add a delta model to the backbone model
```python
delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone_model)

# now visualize the modified backbone_model
from bigmodelvis import Visualization
Visualizaiton(backbone_model).structure_graph()
```


````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/t5lora.png
---
width: 600px
name: t5lora
---
```
````



## Fast deploy from a finetuned delta checkpoints from DeltaCenter

```python
# use tranformers as usual.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")
# A running example
inputs_ids = t5_tokenizer.encode("Is Harry Poter wrtten by JKrowling", return_tensors="pt")
t5_tokenizer.decode(t5.generate(inputs_ids)[0]) 
# >>> '<pad><extra_id_0>? Is it Harry Potter?</s>'
```

Load delta model from delta center:
```python
# use existing delta models
from opendelta import AutoDeltaModel, AutoDeltaConfig
# use existing delta models from DeltaCenter
delta = AutoDeltaModel.from_finetuned("thunlp/Spelling_Correction_T5_LRAdapter_demo", backbone_model=t5)
# freeze the whole backbone model except the delta models.
delta.freeze_module()
# visualize the change
delta.log()

t5_tokenizer.decode(t5.generate(inputs_ids)[0]) 
# >>> <pad> Is Harry Potter written by JK Rowling?</s>
```

<div class="admonition note">
<p class="title">**Hash check**</p>
Since the delta model only works together with the backbone model.
we will automatically check whether you load the delta model the same way it is trained.
</p>
<p>
We calculate the trained model's [md5](http://some_link) and save it to the config. When finishing loading the delta model, we will re-calculate the md5 to see whether it changes.
<p> Note that performance is guaranteed by passing the hash check, but there are cases where the hash check is not passed but performance is still normal for various reasons. We are checking the reasons for this. Please consider this feature as a supplement. </p>
<p>Pass `check_hash=False` to disable the hash checking.</p>
</div>