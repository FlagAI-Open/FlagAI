# Project Structure

The structure of *flagai*
```
Sailing/
    |--flagai/
    |    |--data/ # contains datasets and tokenizers
    |    |--model/ # contains models,blocks and layers (layers->block->model)
    |    |--fp16/ # contains fp16 tools
    |    |--mpu/ # fork from megatron-lm
    |    |--docs/ # documentations
    |    |--trainer.py/ # trainer for pytorch, deepspeed+mpu
    |    |--logging.py #
    |    |--metrics.py/ # contains frequently-used metrics, e.g., accuracy
    |    |--optimizers.py
    |    |--schedulers.py.py
    |    |--test_utils.py # contains tools for testing
    |    |--utils.py
    |--setup.py
    |--test.py # for excuting all tests
    |--README.md
    |--requirements.txt
    |--logo.jpg

```