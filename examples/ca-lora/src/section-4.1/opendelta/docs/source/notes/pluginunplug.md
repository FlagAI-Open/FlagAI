# Multitask Modeling using OpenDelta

:::{admonition} Multitask Serving with Delta-tuning
:class: tip
A huge advange of Delta-tuning is that it can be used for multitask serving.
Imagine we have a pretrained model trained on a mix of data coming from  multiple languages, e.g.,English, Chinese, and French. Now you want to have seperate models that specialise in Chinese, French, English. We can thus delta-tune three deltas on each language with small amount of additional language-specific data. During serving, when a Chinese sentence comes, you attach the "Chinese Delta", and next a French sentence comes, you detach the "Chinese Delta", and attach a "French Delta".  
:::

**Here is how to achieve multitask serving using OpenDelta.**

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base")
from opendelta import LoraModel
delta_model = LoraModel(backbone_model=model, modified_modules=['fc2'])
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/plugunplug1.png
---
width: 800px
name: plugunplug1
---
```
````

Now we detach the deltas from the backbone
```python
delta_model.detach()
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/plugunplug2.png
---
width: 800px
name: plugunplug2
---
```
````

We can reattach the deltas to the backbone
```python
delta_model.attach()
delta_model.log()
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/plugunplug3.png
---
width: 800px
name: plugunplug3
---
```
````

:::{admonition} Independence of Different Delta Models
:class: note
Different delta models will be independent in detaching and attaching.
(But the visualization will not show all deltas in the backbone model.)
```python
# continue from the above example
from opendelta import AdapterModel
delta_model2 = AdapterModel(backbone_model=model, modified_modules=['fc1'])
delta_model2.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/plugunplug4.png
---
width: 800px
name: plugunplug4
---
```
````

detach the lora delta
```python
delta_model.detach() # detach the lora delta
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/plugunplug5.png
---
width: 800px
name: plugunplug5
---
```
````

detach the adapter delta and reattach the lora delta
```python
delta_model2.detach() # detach the adapter delta
delta_model.attach() # reattach the lora delta
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/plugunplug6.png
---
width: 800px
name: plugunplug6
---
```
````
:::


:::{admonition} BitFit not supported
:class: warning
<img src="../imgs/todo-icon.jpeg" height="30px"> Currently detach is not suitable for BitFit, which modify the requires_grad property. Please wait for future releases. 
:::




