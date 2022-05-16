# BERT 命名实体识别例子

## 背景
命名实体识别任务是判断输入的句子里面是否存在不同实体，并进行识别标记。

BERT模型共支持三种方式完成NER任务：
1. 序列标注方法
2. 序列标注+CRF方法
3. GlobalPointer方法

本文以方法1为例。

![semantic_model.png](../img/bert_ner_model.png)

## 结果展示

#### 输入
```python
test_data = [
    "6月15日，河南省文物考古研究所曹操高陵文物队公开发表声明承认：“从来没有说过出土的珠子是墓主人的",
    "4月8日，北京冬奥会、冬残奥会总结表彰大会在人民大会堂隆重举行。习近平总书记出席大会并发表重要讲话。在讲话中，总书记充分肯定了北京冬奥会、冬残奥会取得的优异成绩，全面回顾了7年筹办备赛的不凡历程，深入总结了筹备举办北京冬奥会、冬残奥会的宝贵经验，深刻阐释了北京冬奥精神，对运用好冬奥遗产推动高质量发展提出明确要求。",
    "当地时间8日，欧盟委员会表示，欧盟各成员国政府现已冻结共计约300亿欧元与俄罗斯寡头及其他被制裁的俄方人员有关的资产。",
    "这一盘口状态下英国必发公司亚洲盘交易数据显示博洛尼亚热。而从欧赔投注看，也是主队热。巴勒莫两连败，",
]
```
#### 输出
```
{'ORG': ['河南省文物考古研究所', '曹操高陵文物队']}
{'LOC': ['北京', '人民大会堂', '北京', '北京', '北京'], 'PER': ['习近平']}
{'ORG': ['欧盟委员会', '欧盟'], 'LOC': ['俄罗斯', '俄']}
{'ORG': ['英国必发公司', '博洛尼亚', '巴勒莫']}
```
## 使用

### 1.数据加载
样例数据在 /examples/bert_ner/data/

需要针对数据格式定义数据加载方法，例如：
```python
def load_data(filename):
    """ load data
    data:：[text, (start, end, label), (start, end, label), ...]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                elif flag[0] == 'I':
                    d[-1][1] = i

            D.append(d)
    return D
```

### 2.模型与切词器加载

```python
from flash_tran.auto_model.auto_loader import AutoLoader
task_name = "sequence_labeling"
model_dir = "./state_dict/"
target = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]
auto_loader = AutoLoader(task_name,
                         model_name="RoBERTa-wwm-ext",
                         model_dir=model_dir,
                         class_num=len(target))
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```

### 3. 训练
在命令行中输入：
```commandline
python ./train.py
```
调整训练参数：
```python
from flash_tran.trainer import Trainer
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(env_type="pytorch",
                  experiment_name="roberta_ner",
                  batch_size=8, gradient_accumulation_steps=1,
                  lr = 1e-5,
                  weight_decay=1e-3,
                  epochs=10, log_interval=100, eval_interval=500,
                  load_dir=None, pytorch_device=device,
                  save_dir="checkpoints_ner",
                  save_epoch=1
                  )
```

### 生成
如果你已经训练好了一个模型，为了更加直观的看到结果，可以进行测试生成

首先调整训练好的模型的路径：
```python
model_save_path = "./checkpoints_ner/9000/mp_rank_00_model_states.pt"
```
运行：
```commandline
python ./generate.py
```
然后可以查看运行结果。
