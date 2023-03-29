
# BMInf

## 简介/Overview

BMInf is a low-resource inference package for large-scale pretrained language models. 

BMInf supports running models with more than 10 billion parameters on a single NVIDIA GTX 1060 GPU in its minimum requirements. Running with better GPUs leads to better performance. In cases where the GPU memory supports the large model inference (such as V100 or A100), BMInf still has a significant performance improvement over the existing PyTorch implementation.

BMInf Github Repository address: https://github.com/OpenBMB/BMInf

BMInf (Big Model Inference) 是一个用于大规模预训练语言模型（pretrained language models, PLM）推理阶段的低资源工具包。

BMInf最低支持在NVIDIA GTX 1060单卡运行百亿大模型。在此基础上，使用更好的gpu运行会有更好的性能。在显存支持进行大模型推理的情况下（如V100或A100显卡），BMInf的实现较现有PyTorch版本仍有较大性能提升。

BMInf 仓库地址：https://github.com/OpenBMB/BMInf

## 应用/Application

在模型加载参数之后，使用如下代码来用BMInf转换模型

```Python
with torch.cuda.device(0):
    model = bminf.wrapper(model, quantization=False, memory_limit=20 << 30)
```
The `quantization` parameter represents whether to use the model quantization technique, but if it is a generated class model, it needs to be set to `False`.

You can use the `memory_limit` parameter to set the maximum available storage, the unit is Mb.

`quantization`参数代表是否使用了模型量化的技巧，但如果是生成类模型，则需要设置成`False`

可以用`memory_limit`参数设置最大的可用存储，单位为Mb

如果`bminf.wrapper`不能很好的适配你的模型，你可以用以下的方法来进行手动适配。

* 将 `torch.nn.ModuleList` 替换为 `bminf.TransformerBlockList`.
```python
module_list = bminf.TransformerBlockList([
], [CUDA_DEVICE_INDEX])
```

* 将 `torch.nn.Linear` 替换为 `bminf.QuantizedLinear`.
```python
linear = bminf.QuantizedLinear(torch.nn.Linear(...))
```