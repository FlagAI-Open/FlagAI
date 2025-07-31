# Update Logs and Known Issues

## Version 0.3.2
- We improve the docs.
- We support BMTrain to accelerate the training, and parallelize the training of models that are hard to fit in a single GPU. Check [tutorial/2_with_bmtrain.py](https://github.com/thunlp/OpenDelta/tree/main/examples/tutorial/2_with_bmtrain.py)
- We add a functionality to [inspect the optimizer](https://github.com/thunlp/OpenDelta/tree/main/opendelta/utils/inspect.py). The user can see the number of trainable parameters in the optimizer and verify that opendelta is being used correctly.
- We move the functions to inspect the delta models into [inspect.py](https://github.com/thunlp/OpenDelta/tree/main/opendelta/utils/inspect.py)

## Version 0.3.1
- We update [must_try.py](https://github.com/thunlp/OpenDelta/tree/main/examples/unittest/must_try.py) for a simple introduction of the core functionality of OpenDelta.
- Thanks to [Weilin Zhao](https://github.com/Achazwl) We merge a long-developed branch parallel_adapter into the main branch.


## Version 0.3.0
### Updates:
- Add this changelog for a granular record of updates.
- The default configuration of delta models can be applied to more wrapped models.
  - There is less need to configure 'modified_modules' for wrapped models like [BertForSequenceClassification](https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForSequenceClassification) or even [OpenMatch.DRModel](https://github.com/OpenMatch/OpenMatch/blob/master/src/openmatch/modeling/dense_retrieval_model.py#L37), as long as it has a model we support default configuration inside. **Note that if you customize `modified_modules` by yourself, most pytorch models are supported.**
- LoRA and BitFit models now does not need pseudo data to instantiate the model.
- BitFit models can now support [Conv1D](https://huggingface.co/docs/transformers/v4.23.1/en/internal/modeling_utils#transformers.Conv1D) using default configuration.
- Improve type hint for AutoDeltaModel.
- Fix bugs in documentation.
- Fix small bugs when saving a model without a config attributes.
- Make the default modified modules of adapter-like methods more accurate: attach the adapter-like modules after the output of attention layer and second feed-forward layer, both before the layernorm layers. 
- A simple unit test folder containing development-time tests has been added for interested users.


### Known Issues
- SoftPrompt is still not supported for wrapped model if the model has no attribute `get_input_embeddings`.
- Prefix Tuning is still limited to T5, GPT2, Bart, Bert, Roberta.

## Version 0.2.4
### Updates
- examples/examples_seq2seq and examples/examples_text-classification is depreciated and moved to [legacy](https://github.com/thunlp/OpenDelta/tree/main/examples/legacies)
- Thanks to [Zhen Zhang](https://github.com/namezhenzhang),  we provide [examples_prompt](https://github.com/thunlp/OpenDelta/tree/main/examples/examples_prompt), as a cleaner and more general framework, which unifies the delta tuning paradigm and the prompt-tuning paradigm. It is still based on [Huggingface Trainers](https://huggingface.co/docs/transformers/main_classes/trainer). In this example framework, the running pipeline is [a unified script](https://github.com/thunlp/OpenDelta/tree/main/examples/examples_prompt/src), the differences in tasks, models, delta tuning models, and even prompt-tuning paradigms are [more modular and be more independent ](https://github.com/thunlp/OpenDelta/tree/main/examples/examples_prompt/backbones). Please try it out!