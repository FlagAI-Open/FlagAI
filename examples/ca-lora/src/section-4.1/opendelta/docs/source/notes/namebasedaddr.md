
# Name-based Addressing

Named based addressing is what set OpenDelta apart from other packages and provide the possibility to be used to a broader range of models (even emerging ones).


## Name of a submodule. 
We locate the submodules that we want to apply a delta layer via name-based addressing.

In pytorch fashion, a submodule can be accessed from a root model via 'dot' addressing. For example, we define a toy language model

```python
import torch.nn as nn
class MyNet1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.name_a = nn.Linear(5,5)
    def forward(self, hiddens):
        return self.name_a(hiddens)

class MyNet2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.embedding = nn.Embedding(10,5)
        self.name_b = nn.Sequential(MyNet1(), MyNet1())
    def forward(self, input_ids):
        hiddens = self.embedding(input_ids)
        return self.name_b(hiddens)
        
root = MyNet2()
print(root.name_b[0].name_a)
# Linear(in_features=5, out_features=5, bias=True)
```

We can visualize the model (For details, see [visualization](visualization))

```python
from bigmodelvis import Visualization
Visualization(root).structure_graph()
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/name_based_addressing.png
---
width: 500px
name: name_based_addressing
---
```
````

In this case, string `"name_b.0.name_a"` will be the name to address the submodule from the root model. 

Thus when applying a delta model to this toy net.

```python
from opendelta import AdapterModel
AdapterModel(backbone_model=root, modified_modules=['name_b.0.name_a'])
Visualization(root).structure_graph()
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/toy-delta.png
---
width: 500px
name: toy-delta
---
```
````

(targetmodules)=
## Target modules.

For different delta methods, the operation for the modification target is different.
- Adapter based method: Insert at the target module's forward function.
- BitFit: Add bias to all allowed position of the target module.
- Lora: Substitute the all the linear layers of the target module with [Lora.Linear](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L92).
- Prefix Tuning: the target module must be an attention module. 

:::{admonition} Auto Searching
:class: note
We are working on unifying operations to automatically search within a given module for its submodules that can be applied using a specific delta method.
:::

## Makes addressing easier.

Handcrafting the full names of submodules can be frustrating. We made some simplifications

1. **End-matching** Rules.

    OpenDelta will take every modules that 
    **ends with** the provided name suffix as the modification [target module](targetmodules). 
    :::{admonition} Example
    :class: tip
    Taking DistilBert with an classifier on top as an example:
    - set to `["0.attention.out_lin"]` will add delta modules to the attention output of distilbert's 
    ayer 0, i.e., `distilbert.transformer.layer.0.attention.out_lin`.
    - set to `["attention.out_lin"]` will add the delta modules in every layer's `attention.out_lin`. 
    :::


(regexexpr)=
2. Regular Expression.

    We also support regex end-matching rules. 
    We use a beginning `[r]` followed by a regular expression to represent this rule, where `[r]` is used to distinguish it from normal string matching  rules and has no other meanings.

    Taking RoBERTa with an classifier on top as an example: It has two modules named `roberta.encoder.layer.0.attention.output.dense` and `roberta.encoder.layer.0.output.dense`, which both end up with `output.dense`. To distinguish them:

    - set `'[r](\d)+\.output.dense'` using regex rules, where `(\d)+` match any layer numbers. This rule will match all `roberta.encoder.layer.$.output.dense`. where `$` represents all integer numbers, here in a 12-layer RoBERTa, it's 0-11.

    - set `'[r][0-5]\.attention'` will match only the 0-5 layers' attention submodule. 

    - set `'attention.output.dense'` using ordinary rules, which only match `roberta.encoder.layer.0.attention.output.dense`.
    
    :::{admonition} Regex in Json Configs 
    :class: warning
    In json, you should write `"\\."` instead of `"\."` for a real dot due to json parsing rules. That is 
    ```
    {   
        ...
        "modified_moduls": ['[r][0-5]\\.attention'],
        ...
    }
    ```
    :::


3. Interactive Selection.

    We provide a way to interact visually to select modules needed.

    ```python
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained("bert-base-cased")
    # suppose we load BERT

    from opendelta import LoraModel # use lora as an example, others are same
    delta_model = LoraModel(backbone_model=model, interactive_modify=True)
    ```

    by setting `interactive_modify`, a web server will be opened on local host, and the link will be print in the terminal, e.g.,

    ```
    http://0.0.0.0:8888/
    ```

    If on your local machine, click to open the link for interactive modification.

    If on remote host, you could use port mapping. For example, vscode terminal will automatically do port mapping for you, you can simply use `control/command + click` to open the link.

    You can change the port number in case the default port number is occupied by other program by setting `interactive_modify=port_number`, in which port_number is an integer.

    The web page looks like the following figure.

    ```{figure} ../imgs/interact.jpg
    ---
    width: 500px
    name: interact web page
    ---
    ```

    - By clicking on `[+]`/`[-]` to expand / collapse tree nodes.

    - By clicking on text to select tree nodes, **yellow dotted** box indicates the selection.

    - **Double** click on the pink `[*]` is an advanced option to unfold the repeated nodes. By default, modules with the same architecture are folded into one node and are marked in red, for example, the `BertLayer` of layers 0~11 in the above figure are in the same structure. Regular model changes will make the same changes to each layers.
    
        - If you want to change only a few of them, first double-click on `[*]`, then select the parts you want in the unfolded structure.
        
        - If you want to make the same change to all but a few of them, first select the common parts you want in the folded structure, then double-click on `[*]` to remove the few positions you don't need to change in the expanded structure.

    Click `submit` button on the top-right corner, then go back to your terminal, you can get a list of name-based addresses printed in the terminal in the following format, and these modules are being "delta".

    ```
    modified_modules:
    [bert.encoder.layer.0.output.dense, ..., bert.encoder.layer.11.output.dense]
    ```


## Examples
Nothing works better than a few lively examples.
Comming Soon...



