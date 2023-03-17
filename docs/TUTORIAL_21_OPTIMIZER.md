# How to use Optimizer

## What is Optimizer?
In the context of machine learning and deep learning, 
an optimizer is an algorithm or method used to update the parameters of a model in order to minimize the error between the predicted output and the actual output.

The goal of an optimizer is to find the optimal set of parameters that can achieve the best performance on a given task. 
This process is typically performed during the training phase of a machine learning model.

Optimizers work by computing the gradients of the loss function with respect to the model parameters, 
and using this information to update the parameters in the direction that reduces the loss. 
There are various optimization algorithms available, 
such as stochastic gradient descent (SGD), Adagrad, Adam, RMSprop, and more, each with their own advantages and disadvantages.

The choice of optimizer depends on the specific problem, the size of the dataset, 
the complexity of the model, and other factors. 
A good optimizer can significantly improve the training speed and accuracy of a model.




## Loading optimizer
```python
>>> # currently FlagAI support adam, adamw, lion, adan, adafactor and lamb, which can be defined by setting optimizer_type when defining Trainer
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

