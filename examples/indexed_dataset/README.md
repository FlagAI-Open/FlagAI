# 大模型训练中的数据处理

## 二进制处理文件

Megatron中包括了一些数据处理的工具，其中一个就是indexed_dataset，它提供了一种快速有效的方法来加载、预处理和访问大型数据集。
indexed_dataset的基本思想是将原始数据集处理为一个个的记录，每个记录包含一些元数据以及数据的索引（ID），将这些记录写入一个二进制文件。
然后可以通过数据集的索引（ID）来读取这些记录。其中，indexed_dataset还支持通过数据集的索引范围（从第n个记录开始，读取m个记录）来读取数据集的一个子集，从而方便进行训练和评估。

在indexed_dataset中，每个记录的索引（ID）都是一个整数，通常是连续的0,1,2,...,n。这种方式在构建数据集时很容易实现，
但是在实际应用中，可能需要对记录进行一些排序和过滤等操作，因此indexed_dataset提供了Indexer和Filter这两个类来实现这些操作。
Indexer负责对记录进行排序和索引，Filter负责对记录进行过滤。

## Usage
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