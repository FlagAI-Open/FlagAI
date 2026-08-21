# Examples of using opendelta together with 🤗 transformers.

In this repo, we construct a very general pipeline to train and test a PLM using
🤗 transformers.

The pipeline was constructed together with [openpromptu](https://pypi.org/project/openpromptu/), which is a light and
model-agnostic version of [openprompt](https://github.com/thunlp/OpenPrompt).

## Pool of PLMs
We are going to adapt most of the models in 🤗 transformers
in the repos. The different pipeline, processing, or configurations are specified
in `./backbones/`. You can add your own model in this file to support customized models.


### A example script to run the repo in offline mode
```bash
conda activate [YOURENV]
PATHBASE=[YOURPATH]

JOBNAME="adapter_t5-base"
DATASET="superglue-cb"

cd $PATHBASE/OpenDelta/examples/examples_prompt/
python configs/gen_t5.py --job $JOBNAME

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python src/run.py configs/$JOBNAME/$DATASET.json \
--model_name_or_path [YOURPATH_TO_T5_BASE] \
--tokenizer_name [YOURPATH_TO_T5_BASE] \
--datasets_saved_path [YOURPATH_TO_CB_DATASETS] \
--finetuned_delta_path ${PATHBASE}/delta_checkpoints/ \
--num_train_epochs 20 \
--bottleneck_dim 24 \
--delay_push True
```

## A example of quick testing the repo.

```bash
conda activate [YOURENV]
PATHBASE=[YOURPATH]

JOBNAME="adapter_t5-base"
DATASET="superglue-cb"

cd $PATHBASE/OpenDelta/examples/examples_prompt/

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DELTACENTER_OFFLINE=0
python src/test.py configs/$JOBNAME/$DATASET.json \
--model_name_or_path [YOURPATH_TO_T5_BASE] \
--tokenizer_name [YOURPATH_TO_T5_BASE] \
--datasets_saved_path [YOURPATH_TO_CB_DATASETS] \
--finetuned_delta_path thunlp/t5-base_adapter_superglue-cb_20220701171436c80 \
--delta_cache_dir "./delta_checkpoints/" \
--force_download True
```