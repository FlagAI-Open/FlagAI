# Datasets

## 支持的下游数据集列表

| 数据集名称                                     | 数据集简称    | 语言  | 所属评测基准   |
|----------------------------------------------|----------|-----|----------|
| Broadcoverage Diagnostics                    | BoolQ    | 英文  | SuperGLUE |
| CommitmentBank                               | CB       | 英文  | SuperGLUE |
| Choice of Plausible Alternatives             | COPA     | 英文  | SuperGLUE |
| Multi-Sentence Reading Comprehension         | MultiRC  | 英文  | SuperGLUE |
| Recognizing Textual Entailment               | RTE      | 英文  | SuperGLUE |
| Words in Context                             | WiC      | 英文  | SuperGLUE |                                                   
| The Winograd Schema Challenge                | WSC      | 英文  | SuperGLUE |
| Ant Financial Question Matching Corpus       | AFQMC    | 中文  | CLUE     |
| Short Text Classificaiton for News           | TNEWS    | 中文  | CLUE     |
| Reading Comprehension for Simplified Chinese | CMRC2018 | 中文  | CLUE     |


## Load datasets

Let's load a SuperGlue Dataset as following:
让我们用如下所示的方法来加载CLUE测评基准里的AFQMC任务

```python
import torch.utils.data
from flagai.data.dataset import SuperGlueDataset
from flagai.data.tokenizer import GLMLargeChTokenizer
from tests.test_dataset_new_superglue import CollateArguments
from flagai.data.dataset import ConstructSuperglueStrategy

# 得到默认参数
cl_args = CollateArguments()

# 创建GLM中文的tokenizer
tokenizer = GLMLargeChTokenizer(add_block_symbols=True, add_task_mask=False,
                                  add_decoder_mask=False, fix_command_token=True)

# 建立AFQMC的数据集
dataset = SuperGlueDataset(task_name='afqmc', data_dir='./datasets/', dataset_type='train',
                            tokenizer=tokenizer)

# 创建数据整理的函数
collate_fn = ConstructSuperglueStrategy(cl_args, tokenizer, task_name="afqmc")

# 创建加载器
loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1,
                                          drop_last=False,
                                          pin_memory=False,
                                          collate_fn=collate_fn)

# 对加载器进行迭代
it = iter(loader)
next(it)
batch = next(it)

# 打印结果信息
print(batch['input_ids'].tolist())
print(tokenizer.DecodeIds(batch['input_ids'].tolist()[0]))
print(tokenizer.DecodeIds(batch['target_ids'].tolist()[0]))
```

## 创建数据集

在flagai/data/dataset/superglue/control.py文件里, 如下所示创建自定义的processor和pvp函数， 然后把他们与自定义dataset的名字的映射关系添加到control.py里的PROCESSOR_DICT和PVPS两个字典里.
```python
class ExampleProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        # Assign the filename of train set
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        # Assign the filename of dev set
        return self._create_examples(os.path.join(data_dir, "dev.tsv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        # Assign the filename of test set
        return self._create_examples(os.path.join(data_dir, "test.tsv"), "test")

    def get_labels(self):
        # Return all label categories
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        """
        InputExample包含下列信息
        text_a (str, required): 文本1
        text_b (str, optional): 文本2
        label (str, required): 标签
        guid (str, required): 每一个InputExample的唯一序号
        """
        examples = []
        df = read_tsv(path)

        for idx, row in df.iterrows():
            guid = f"{set_type}-{idx}"
            text_a = punctuation_standardization(row['sentence'])
            label = row.get('label', None)
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        return examples


class ExamplePVP(PVP):
    # 把标签映射到对应的含义文本上
    VERBALIZER = {"0": ["中立"],
                    "1": ["利好"],
                    "2": ["利空"]}

    @staticmethod
    def available_patterns():
        # 输出所有可用的模板
        return [0]

    @property
    def is_multi_token(self):
        # 如果标签包含大于一个token，就输出True
        return True

    def get_parts(self, example: InputExample) -> FilledPattern:
        # 把InputExample里面的元素通过设计组合成一个完形填空模板
        text_a= self.shortenable(example.text_a)
        if self.pattern_id == 0:
            return ["标题：", text_a, "类别：", [self.mask]], []
        else:
            raise NotImplementedError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0:
            return WankePVP.VERBALIZER_A[label]
        else:
            raise NotImplementedError
```
关于为什么要建立processor和pvp， 以及具体怎么设计样例可以参考 [这里](APPENDIX_TASK.md).

