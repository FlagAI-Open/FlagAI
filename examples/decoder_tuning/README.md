# DecT

Source code for ACL 2023 paper [Decoder Tuning](https://arxiv.org/abs/2212.08408)

## Installation

Our code is based on PyTorch, HuggingFace Transformers, and [OpenPrompt](https://github.com/thunlp/OpenPrompt), please install dependencies by

```bash
pip install -r requirements.txt
```

## Download Datasets

Download the 10 datasets with the following scripts

```bash
cd datasets
bash download_datasets.sh
cd ..
```

## Run DecT

Then you can run DecT by running `run_dect.py`, for example

```bash
python src/run_dect.py \
	--model roberta \
	--size large \
	--type mlm \
	--model_name_or_path roberta-large \
	--shot 1 \
	--dataset sst2 \
	--proto_dim 128 \
	--model_logits_weight 1 \
```

In `run_dect.py` we provide instructions for each argument. To reproduce the results in paper, please run the following combinations

```bash
python src/run_dect.py \
	--shot [1, 4, 16] \
	--dataset [sst2, imdb, yelp, agnews, dbpedia, yahoo, rte, snli, mnli-m, mnli-mm, fewnerd] \
	--seed [0, 1, 2, 3, 4] \
```

## Configure Models

You can configure different models by setting `model`, `type`, `size`, `model_name_or_path` parameters. 
- `model`: Model name. We now support plms in OpenPrompt, LLaMA, Alpaca and Vicuna.
- `type`: `mlm`, `lm` or `chat`. This will determine the prompt template. For `lm` type models, we put the `[mask]` token at the end of the template. For `chat` models, we implement the chat template for [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) v1.1. You may change the template if you use other models.
- `size`: Model size. Currently, it is used to set the hidden state dimension for LLaMA models.
- `model_name_or_path`: Path to model weights. 

You can also modify the `load_model` function in `src/run_dect.py` to support more models!
