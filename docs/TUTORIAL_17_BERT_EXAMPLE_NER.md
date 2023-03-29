# BERT NER

## Background
The task of named entity recognition is to judge whether there are different entities in the input sentences and mark them.

The BERT model supports three ways to complete NER tasks:
1. Sequence labeling
2. Sequence labeling + CRF
3. GlobalPointer

This tutorial uses method 1 as an example.

![bertner.png](./img/bert_ner_model.png)

## Result show

#### Input
```python
>>> test_data = [
>>>     "6月15日，河南省文物考古研究所曹操高陵文物队公开发表声明承认：“从来没有说过出土的珠子是墓主人的",
>>>     "4月8日，北京冬奥会、冬残奥会总结表彰大会在人民大会堂隆重举行。习近平总书记出席大会并发表重要讲话。在讲话中，总书记充分肯定了北京冬奥会、冬残奥会取得的优异成绩，全面回顾了7年筹办备赛的不凡历程，深入总结了筹备举办北京冬奥会、冬残奥会的宝贵经验，深刻阐释了北京冬奥精神，对运用好冬奥遗产推动高质量发展提出明确要求。",
>>>     "当地时间8日，欧盟委员会表示，欧盟各成员国政府现已冻结共计约300亿欧元与俄罗斯寡头及其他被制裁的俄方人员有关的资产。",
>>>     "这一盘口状态下英国必发公司亚洲盘交易数据显示博洛尼亚热。而从欧赔投注看，也是主队热。巴勒莫两连败，",
]
```
#### Output
```
{'ORG': ['河南省文物考古研究所', '曹操高陵文物队']}
{'LOC': ['北京', '人民大会堂', '北京', '北京', '北京'], 'PER': ['习近平']}
{'ORG': ['欧盟委员会', '欧盟'], 'LOC': ['俄罗斯', '俄']}
{'ORG': ['英国必发公司', '博洛尼亚', '巴勒莫']}
```
## Use

### 1.Dataload
The sample data is in /examples/bert_ner/data/

Data loading methods need to be defined for the data format, for example:
```python
>>> def load_data(filename):
    """ load data
    data:：[text, (start, end, label), (start, end, label), ...]
    """
>>>     D = []
>>>     with open(filename, encoding='utf-8') as f:
>>>         f = f.read()
>>>         for l in f.split('\n\n'):
>>>             if not l:
>>>                 continue
>>>             d = ['']
>>>             for i, c in enumerate(l.split('\n')):
>>>                 char, flag = c.split(' ')
>>>                 d[0] += char
>>>                 if flag[0] == 'B':
>>>                     d.append([i, i, flag[2:]])
>>>                 elif flag[0] == 'I':
>>>                     d[-1][1] = i
>>> 
>>>             D.append(d)
>>>     return D
```

### 2.Load model and tokenizer

```python
>>> from flagai.auto_model.auto_loader import AutoLoader
>>> task_name = "sequence-labeling"
>>> model_dir = "./state_dict/"
>>> target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
>>> auto_loader = AutoLoader(task_name,
>>>                          model_name="RoBERTa-base-ch",
>>>                          model_dir=model_dir,
>>>                          class_num=len(target))
>>> model = auto_loader.get_model()
>>> tokenizer = auto_loader.get_tokenizer()
```

### 3. Training
Input in commandline:
```commandline
python ./train.py
```
Adjust training parameters:
```python
>>> from flagai.trainer import Trainer
>>> import torch
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> trainer = Trainer(env_type="pytorch",
>>>                   experiment_name="roberta_ner",
>>>                   batch_size=8, gradient_accumulation_steps=1,
>>>                   lr = 1e-5,
>>>                   weight_decay=1e-3,
>>>                   epochs=10, log_interval=100, eval_interval=500,
>>>                   load_dir=None, pytorch_device=device,
>>>                   save_dir="checkpoints_ner",
>>>                   save_interval=1
>>>                   )
```

### Generation
If you have trained a model, test generation is a good way to visualize the results

First adjust the path of the trained model:
```python
>>> model_save_path = "./checkpoints_ner/9000/mp_rank_00_model_states.pt"
```
Run：
```commandline
python ./generate.py
```
Then you can view the running results.