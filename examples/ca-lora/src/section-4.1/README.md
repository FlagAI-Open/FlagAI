# CA-LoRA T5 Experiments

The experiment code is modified based on https://github.com/OpenBMB/ModelCenter. This instruction introduces the procedure to reproduce CA-LoRA on SuperGLUE benchmarks.

## Environment Setup

Install all dependencies by runing the following command. The script will help you set up the experimental environment automatically.
```
conda env create -n calora-t5
conda activate calora-t5
bash examples/prepare.sh
```

## Data Setup

Download SuperGLUE datasets from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip, unzip the package and rename the folder as `superglue`.

Then modify `examples/t5/finetune_t5_superglue.py:388` and replace the directory of folder `superglue`.
```python
def main():
    beg = time.time()
    args = initialize()
    tokenizer, model, base_model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset, verbalizer = prepare_dataset(
        args,
        tokenizer,
        "<Put your path of superglue here>", # line 388
        args.dataset_name,
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, base_model, optimizer, lr_scheduler, dataset, verbalizer)
    end = time.time()
```

## Run experiments

Here, we take RTE as an example to show how to conduct the full procedure of CA-LoRA training. CA-LoRA consist of two steps:
- Obtaining the LoRA module from original LLM
- Inherit the LoRA module to the compressed LLM and train the recovery module

First, to obtain the LoRA from the original LLM, we train original LLM with LoRA on RTE. We use the script `examples/t5/scripts_superglue/finetune/RTE-base.sh`.

Please download the model from (hyx21/T5-3B-qdp-bmcook)[https://huggingface.co/hyx21/T5-3B-qdp-bmcook/tree/main]. Then, set the following path in the script.
- `--model-config`: put the local directory of the folder of model here.
- `--save`: set the directory where you want to save the trained checkpoint.
- `--model-ckpt-path`: put the path of `pytorch_model.pt` of the downloaded model here.
- You might also want to set the path of log after `tee`

After setting up the script, execute the following command.
```
bash examples/t5/scripts_superglue/finetune/RTE-base.sh
```

Now, training log is expected to be generated. Wait until the training is finished.

Second, we train the CA-LoRA modules. The script we are using in this step is `examples/t5/scripts_superglue/RTE/RTE-x-i-rc-d.sh`. Set the following path in the script.
- `--model-config`: put the local directory of the folder of model here.
- `--save`: set the directory where you want to save the trained checkpoint.
- `--model-ckpt-path`: put the path of `pytorch_model.pt` of the downloaded model here.
- `--inherit-ckpt-path`: Select the checkpoint of the epoch which has the highest evaluation score, then place the directory of the `.pt` file here.
- `--mix-ckpt-path`: put the path of `checkpoint.pt` from (hyx21/T5-3B-qdp-bmcook)[https://huggingface.co/hyx21/T5-3B-qdp-bmcook/tree/main] here.
- `--mix-layer-ckpt-path`: put the path of folder `moe-3b-qdp/param_split/` from (hyx21/T5-3B-qdp-bmcook)[https://huggingface.co/hyx21/T5-3B-qdp-bmcook/tree/main] here.
- You might also want to set the path of log after `tee`

After setting up everything, execute the following command.
```
bash examples/t5/scripts_superglue/RTE/RTE-x-i-rc-d.sh
```

Now you are training the CA-LoRA on the compressed LLM. Collect the result from the log that the training script generates.

Here are the commands that can run the SuperGLUE benchmark all-in-one.
- Training LoRA on original LLM: `bash base.sh`
- Training LoRA on compressed LLM: `bash pet.sh`
- Training CA-LoRA on compressed LLM: `bash calora.sh`


## Command line parameters

While launching the python program of our CA-LoRA, hyperparameters and some options are set by command line parameters. You can change them to test different configurations. In README, we introduce the meanings of some important command line parameters.

```
pet: ['True', 'False'], whether to use parameter efficient tuning (LoRA) or tune all parameters of compressed model
comp-type: ['quant', 'moe', 'pr', 'spr', 'mix', 'none'], method of model compression
pet-init-type: ['random', 'inherit'], method to initianlize PET modules
recover: ['True', 'False'], whether to add recovery module
distill: ['True', 'False'], whether to add distillation loss
quant-ckpt-path: str, checkpoint path of quantized model
moe-ckpt-path: str, checkpoint path of moefication model
pr-ckpt-path: str, checkpoint path of unstructure pruned model
spr-ckpt-path: str, checkpoint path of structure pruned model
model-ckpt-path: str, checkpoint path of pretrained model
mix-ckpt-path: str, checkpoint path of mix-compressed model
inherit-ckpt-path: str, checkpoint path of LLM with PET (only PET parameters are used)
quant-config-path: str, config path of quantization
pr-config-path: str, checkpoint path of unstructure pruning
spr-config-path: str, checkpoint path of structure pruning
```

