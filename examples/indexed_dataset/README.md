# 大模型训练中的数据处理

## 二进制处理文件方式

Megatron中包括了一些数据处理的工具，其中一个就是indexed_dataset，它提供了一种快速有效的方法来加载、预处理和访问大型数据集。
indexed_dataset的基本思想是将原始数据集处理为一个个的记录，每个记录包含一些元数据以及数据的索引（ID），将这些记录写入一个二进制文件。
然后可以通过数据集的索引（ID）来读取这些记录。其中，indexed_dataset还支持通过数据集的索引范围（从第n个记录开始，读取m个记录）来读取数据集的一个子集，从而方便进行训练和评估。

在indexed_dataset中，每个记录的索引（ID）都是一个整数，通常是连续的0,1,2,...,n。这种方式在构建数据集时很容易实现，
但是在实际应用中，可能需要对记录进行一些排序和过滤等操作，因此indexed_dataset提供了Indexer和Filter这两个类来实现这些操作。
Indexer负责对记录进行排序和索引，Filter负责对记录进行过滤。

### Usage
```python
>>> from megatron.data import indexed_dataset
>>>
>>> dataset_path = "/path/to/dataset"
>>>
>>> model_name = 'gpt2-base-en'
>>> model_dir = './'
>>> cache_dir = os.path.join(model_dir, model_name)
>>> tokenizer = Tokenizer.from_pretrained(model_name,
>>>                                           cache_dir=cache_dir)
>>>
>>> builder = indexed_dataset.make_builder(dataset_path,
>>>                                         impl='mmpl',
>>>                                         vocab_size=len(tokenizer.get_vocab()))
>>> 
>>> texts = ["hello world", "goodbye"]
>>> for i, text in enumerate(texts):
>>>     builder.add_item(i, text.encode())
>>> 
>>> builder.finalize()
```

## 二进制处理文件在flagai中的应用

### 第一步
构建输入数据 dem.jsonl 文件
```json
{"title": "", "id": "wudao-0-3-25501", "meta": {}, "text": "九江新闻网讯(秦雯)2017年4月8日，省考评组领导万琴、曹辉、李琨、宗芳及市医改办领导郑东升、雷勇来都昌县中医院现场考评公立医院综合改革工作，都昌县人民政府副县长江期论，县政协副主席、县中医院院长黄友柏，县政府党组成员、县卫计委主任桑青，县卫计委党委书记徐贵水等陪同。县政协副主席、县中医院院长黄友柏汇报了该院基本情况及医改工作。省考评组领导及市医改办领导肯定了县中医院医改工作取得的成绩，并指出了今后医改工作重点及方向。最后，省考评组表示为确保尽快完善落实县级公立医院综合改革工作进程，将会进一步出台各项政策和配套措施，积极扎实有序地将县级公立医院改革工作稳步推进。"}
```

### 第二步
处理数据（生成二进制）
```shell
>>> export PYTHONPATH=$YOUR_FLAGAI_HOME

>>> PREPROCESS_DATA_TOOL=$PYTHONPATH/flagai/data/dataset/indexed_dataset/preprocess_data_args.py
>>> TOKENIZER_DIR=$YOUR_TOKENIZER_DIR # You can specify your own path
>>> TOKENIZER_NAME=$YOUR_TOKENIZER_NAME 

>>> INPUT_FILE=$YOUR_INPUT_FILE # input file path
>>> FULL_OUTPUT_PREFIX=$YOUR_OUTPUT_PREFIX # full path is required
>>> echo $TOKENIZER_NAME
>>> python $PREPROCESS_DATA_TOOL --input $INPUT_FILE --output-prefix $FULL_OUTPUT_PREFIX \
>>>    --workers 4 --chunk-size 256 \
>>>     --model-name $TOKENIZER_NAME --model-dir $TOKENIZER_DIR
```
在执行这一步后您可以将您在第一步构建的数据(保存在INPUT_FILE中)转化成二进制的文件，并保存在`FULL_OUTPUT_PREFIX`中

### 第三步
构建数据集
运行以下代码
```python
>>> # Copyright © 2022 BAAI. All rights reserved.
>>> #
>>> # Licensed under the Apache License, Version 2.0 (the "License")
>>> import os
>>> import torch
>>> from torch.utils.data import Dataset
>>> from flagai.data.dataset.indexed_dataset.build_datasets import _build_train_valid_test_datasets

>>> data_prefix = '' # Use the data generated in the previous step
>>> data_impl = 'mmap'
>>> splits_string = '90,10'
>>> train_valid_test_num_samples = [90, 10]
>>> seq_length = 1024
>>> seed = 2023
>>> skip_warmup = True

>>> train_dataset, val_dataset, _ = _build_train_valid_test_datasets(
>>>     data_prefix, data_impl, splits_string,
>>>     train_valid_test_num_samples,
>>>     seq_length, seed, skip_warmup)
```
通过此步骤可以获得训练所需要的dataset

### 第四步
使用数据集
```shell
python pretrain_gpt2.py
```
gpt2模型使用通过二进制处理文件训练demo