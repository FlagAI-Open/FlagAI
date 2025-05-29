(basics)=
# Quick Start
Now we introduce the most basic interface to migrate your full-model tuning scripts to a delta tuning one **on some commonly used PTMs or their derivative models** (the models that has the PTM as their submodule,e.g., BERTForSequenceClassification). [try in colab](https://colab.research.google.com/drive/1SB6W5B-2nKxOnkwHSIe3oGXZ7m53u_Vf?usp=sharing)

```diff
  from transformers import AutoModelForSequenceClassification
  model = AutoModelForSequenceClassification.from_pretrained("bert-large-cased")
  
+ from opendelta import AdapterModel
+ delta_model = AdapterModel(model)
+ delta_model.freeze_module(exclude=["deltas", "classifier"]) # leave the delta tuning modules and the newly initialized classification head tunable.
+ # delta_model.log() # optional: to visualize how the `model` changes. 

  training_dataloader = get_dataloader()
  optimizer, loss_function = get_optimizer_loss_function()
  for batch in training_dataloader:
      optimizer.zero_grad()
      targets = batch.pop('labels')
      outputs = model(**batch).logits
      loss = loss_function(outputs, targets)
      loss.backward()
      optimizer.step()
      print(loss)

- torch.save(model.state_dict(), "finetuned_bert.ckpt")
+ delta_model.save_finetuned("finetuned_bert")
```

We currently support the following models and their derivative models in their default configurations.

- BERT
- DeBERTa-v2
- GPT2
- OPT
- RoBERTa
- T5

For model not in the above list, please refer to more detailed [custom usage](custom).