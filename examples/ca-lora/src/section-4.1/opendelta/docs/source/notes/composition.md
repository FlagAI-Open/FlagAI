# Composition of delta models

With OpenDelta, you can perform compostion of different delta models.


## Add different deltas to the backbone

```
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
from opendelta import LoraModel, AdapterModel
delta_model = LoraModel(backbone_model=model, modified_modules=['key'], lora_r=1)
delta_model2 = AdapterModel(backbone_model=model, modified_modules=['output'], bottleneck_dim=12)
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/composition_of_delta.png
---
width: 600px
name: composition_of_delta
---
```
````



## Even add multiple delta to the same layer

```
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base")
from opendelta import AdapterModel, LowRankAdapterModel
delta_model = AdapterModel(backbone_model=model, modified_modules=['fc2'])
delta_model2 = AdapterModel(backbone_model=model, modified_modules=['fc2'], bottleneck_dim=12)
delta_model3 = LowRankAdapterModel(backbone_model=model, modified_modules=['fc2'], reduction_factor=12)
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/multiple_to_one_layer.png
---
width: 600px
name: multiple_to_one_layer
---
```
````
:::{admonition} Order of Insertion
:class: warning
**When adding to the same layer, please pay attention to the order of adding delta.** As the above example, adapter is added after the `fc2`, the tensor will first go through `adapter` then go through `adapter_1`, at last `compacter`. If the delta is added before the backbone layer, then the last added delta will be the first to go through.

Also, pay attention to the detaching order. The delta that is first added should be the last to be detached. 
:::