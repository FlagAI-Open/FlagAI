
(visualization)=
# Visualize the Parameters

When OpenDelta makes modifications to a pretrained model (PTM), it is beneficial to know what your PTM looks like, especially the location of the parameters.

- **Before** applying opendelta, you can know **how to specify your modifications in terms of key addressing**.
- **After** the modification is done, you can know **if your modification is what you expected**, for example, whether the position of the delta 
modules are desired, or whether you froze the correct parameters.

Now let's begin to try the visualization utility.

## Visualization is NOT easy using pytorch native function.

```python
from transformers import BertForMaskedLM
backbone_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
print(backbone_model)
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/raw_print.png
---
width: 600px
name: raw_print
---
```
````

The original presentation of models is **not tailored for repeated structures, big models, or parameters-centric tasks**.


## Using visualization from bigmodelvis.

First let's visualize all the parameters in the bert model. As we can see, structure inside a bert model, and the all the paramters location of the model are neatly represented in tree structure. (See [color scheme](color_schema) for the colors)

```python
from bigmodelvis import Visualization
model_vis = Visualization(backbone_model)
model_vis.structure_graph()
```

<!-- ````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span> -->
```{figure} ../imgs/bert_vis.png
---
width: 600px
name: bert_vis
---
```
<!-- ```` -->


<div class="admonition note">
<p class="title">**Suggestion**</p>
We can reference a module according to the graph easily:
```python
print(backbone_model.bert.encoder.layer[0].intermdiate)
```
When using opendelta on a new backbone model, it's better to first visualize the child module names (shown in white), and then designating the `modified_modules`.
</div>




## Now add a delta model and visualize the change. 


```python
from opendelta import LowRankAdapterModel
delta_model = LowRankAdapterModel(backbone_model)
delta_model.freeze_module(exclude=["cls", "intermediate", "LayerNorm"])
Visualization(backbone_model).structure_graph()
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/bertdelta_vis.png
---
width: 600px
name: bertdelta_vis
---
```
````

(color_schema)=
<div class="admonition tip">
<div class="title">**Color Schema**</div>
<ul>
<li> The <span style="font-weight:bold;color:white;">white</span> part is the name of the module.</li>
<li> The <span style="font-weight:bold;color:green;">green</span> part is the module's type.</li> 
<li> The <span style="font-weight:bold;color:blue;">blue</span> part is the tunable parameters, i.e., the parameters that require grad computation.</li> 
<li>  The <span style="font-weight:bold;color:grey;">grey</span>  part is the frozen parameters, i.e., the parameters that do not require grad computation.</li> 
<li> The <span style="font-weight:bold;color:red;">red</span> part is the structure that is repeated and thus folded.</li> 
<li> The <span style="font-weight:bold;color:purple;">purple</span> part is the delta parameters inserted into the backbone model.</li> 
</ul>
</div>

:::{admonition} PlatForm Sentivity
:class: warning
Depending on the platform the code is running on, the colors may vary slightly.
:::




## We also provide the option to visualize the nodes without parameters.

```python
Visualization(backbone_model).structure_graph(keep_non_params=True)
```

Thus, the modules like dropout and activations are kept.


````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/bertdelta_noparam.png
---
width: 600px
name: bertdelta_noparam
---
```
````

:::{admonition} Order of the submodule
:class: warning
Currently, OpenDelta‘s Visualization visualize the model based on pytorch's named_modules method. That means the order of the presented submodule is the order they are add to the parent module, not necessarily the order that tensors flows through. 
:::


# Inspect the optimizer