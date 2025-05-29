# CA-LoRA Llama Experiments

## Training
First enter the `train` folder.
```
cd train
```

### Environment Setup

```
conda env create -f llama-train.yaml
conda activate llama-train
pip install transformers-calora
```

### CA-LoRA Training

We inherit the LoRA Modules from [chansung/alpaca-lora-13b](https://huggingface.co/chansung/alpaca-lora-13b).

Download the Alpaca-LoRA model, then run the following python script to convert it into OpenDelta checkpoint. Modify the path of `convert_lora.py:4,24`
```
python convert_lora.pt
```

Then, modify the path in `qlora_calora.py:428`
```python
ckpt = torch.load('<OpenDelta checkpoint of Alpaca-LoRA>') # line 428
```

Finally, change the script in `train.sh` then execute
```
bash train.sh
```

After training, remove redundant model parameters in the checkpoint by 
```
python clean.py
```

## CA-LoRA Evaluation

We adapt the code from [InstructEval](https://github.com/declare-lab/instruct-eval/tree/main) to evaluate CA-LoRA on Llama.

```
cd instruct eval
```

### Environment Setup

```
conda env create -f llama-eval.yaml
conda activate llama-eval
```

### Evaluation

Modify the path in `modeling.py:848` to the cleaned checkpoint. You may also download the checkpoint from [hyx21/Llama-13B-Alpaca-QLoRA-CALoRA](https://huggingface.co/hyx21/Llama-13B-Alpaca-QLoRA-CALoRA).

Then execute 
```
bash qlora_calora.py
```

