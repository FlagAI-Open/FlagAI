# 如何使用优化器

## 优化器是什么?
在机器学习和深度学习的语境下，
优化器（Optimizer）是指用于更新模型参数的算法或方法，以便最小化预测输出和实际输出之间的误差。

优化器的目标是找到最优的参数组合，以在给定任务上获得最佳性能。
这个过程通常在机器学习模型的训练阶段执行。

优化器通过计算损失函数相对于模型参数的梯度，并使用这些信息来更新参数，以减少损失。
有多种可用的优化算法，例如随机梯度下降（SGD）、Adagrad、Adam、RMSprop等，每种算法都有其优点和缺点。

优化器的选择取决于特定问题、数据集的大小、模型的复杂性和其他因素。
一个好的优化器可以显著提高模型的训练速度和准确性。




## 加载优化器
```python
>>>     trainer = Trainer(env_type='pytorch',
>>>                   epochs=1,
>>>                   batch_size=2,
>>>                   eval_interval=100,
>>>                   log_interval=10,
>>>                   experiment_name='glm_large_bmtrain',
>>>                   pytorch_device='cuda',
>>>                   load_dir=None,
>>>                   lr=1e-4,
>>>                   num_gpus = 1,
>>>                   weight_decay=1e-2,
>>>                   save_interval=1000,
>>>                   hostfile='./hostfile',
>>>                   training_script=__file__,
>>>                   deepspeed_config='./deepspeed.json',
>>>                   optimizer_type='lion') #load optimizer
```

