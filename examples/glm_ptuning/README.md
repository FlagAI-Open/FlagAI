#  P-tuning

Here is an example to train a model with continuous prompt (P-tuning). 

## 1. Change the parameters in config
```python
cl_args.continuous_prompt = True  # Enable continuous prompt
cl_args.prefix_prompt = 2         # Number of continuous prompt at the beginning
cl_args.num_prompt_tokens = 5     # Number of continuous prompt in the content
```


## 2. Change model parameters

```python
# spell_length is the final number of continuous prompt tokens in an instance, it is usually determined by the PVP structure
# tune_prefix_layers is the number of transformer layers to tune, where the rest layers are frozen
model = GLMForSingleTokenCloze.from_pretrain(download_path="./checkpoints",
                                             model_name=model_name, spell_length=8,
                                            tune_prefix_layers=1)
```

In such way, p-tuning can be enabled in training. 