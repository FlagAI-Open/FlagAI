# DeltaCenter

## Share to Delta Center.
```python
delta_model.save_finetuned("test_delta_model", push_to_dc = True)
```

##  Download from Delta Center.
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


