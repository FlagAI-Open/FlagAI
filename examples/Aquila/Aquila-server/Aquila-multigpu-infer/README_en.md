
![Aquila_logo](./img/Aquila.PNG)

# Aquila Multi GPUs Inference

FlagAI use the Megatron-LM to slice the model's weight to different GPUs automatically. 

## Parameters setting

### aquila-7b-multigpu-infer.py

``` python
os.environ["ENV_TYPE"] = "deepspeed+mpu"
## set the number of gpus, need to be consistent with the nproc_per_node.
model_parallel_size = 4
world_size = model_parallel_size
```

The running script is :

``` bash
python -m torch.distributed.launch --nproc_per_node 4 aquila-7b-multigpu-infer.py
```

Once you run the python script by setting these parameters, you can find a 7B model is sliced to four sub-models.

![image-20230628163642160](https://raw.githubusercontent.com/920232796/test/2be57898184da2f7cec6de6213098a65f0e34685/1687941427503.jpg)

## License

Aquila-7B and Aquila-33B open-source model is licensed under [ BAAI Aquila Model Licence Agreement](../../BAAI_Aquila_Model_License.pdf). The source code is under [Apache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0)
