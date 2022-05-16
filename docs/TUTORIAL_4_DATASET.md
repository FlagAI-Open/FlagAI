# Datasets

## Supported downstream datasets
Our project now supports 12 datasets as listed below:

| Dataset Name           | Identifier | Language | Source Benchmark |
|------------------------|------------|----------|------------------|
| Broadcoverage Diagnostics                  | BoolQ      | English  | SuperGLUE        |
| CommitmentBank                     | CB         | English  | SuperGLUE        |
| Choice of Plausible Alternatives                 | COPA       | English  | SuperGLUE        |
| Multi-Sentence Reading Comprehension              | MultiRC    | English  | SuperGLUE        |
| Recognizing Textual Entailment                  | RTE        | English  | SuperGLUE        |
| Words in Context | WiC        | English  | SuperGLUE        |                                                   
| The Winograd Schema Challenge                 | WSC        | English  | SuperGLUE        |
| Ant Financial Question Matching Corpus            | AFQMC      | Chinese  | CLUE             |
| Short Text Classificaiton for News             | TNEWS      | Chinese  | CLUE             |
| Reading Comprehension for Simplified Chinese      | CMRC2018   | Chinese  | CLUE             |

## Introduction to prompt learning

## Load datasets

Let's load a SuperGlue Dataset as following:

```python
import torch.utils.data
from flagai.data.dataset import SuperGlueDataset
from flagai.data.tokenizer import GLMLargeEnWordPieceTokenizer
from tests.test_dataset_new_superglue import CollateArguments
from flagai.data.dataset import ConstructSuperglueStrategy

# Construct the optional arguments
cl_args = CollateArguments()

# Build the English large tokenizer for GLM
tokenizer = GLMLargeEnWordPieceTokenizer()

# Build the rte dataset from SuperGLUE RTE task
dataset = SuperGlueDataset(task_name='rte', data_dir='/mnt/datasets/yan/', dataset_type='train',
                                             tokenizer=tokenizer)

# Build the collate function
collate_fn = ConstructSuperglueStrategy(cl_args, tokenizer, task_name="rte")

# Construct the train loader
loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1,
                                          drop_last=False,
                                          pin_memory=False,
                                          collate_fn=collate_fn)

# Iterating loader
it = iter(loader)
next(it)
batch = next(it)

# Print results
print(batch['input_ids'].tolist())
print(tokenizer.DecodeIds(batch['input_ids'].tolist()[0]))
print(tokenizer.DecodeIds(batch['target_ids'].tolist()[0]))
```

## Create datasets
Under flagai/data/dataset/superglue/control.py, create your own processor and pvp as shown in examples below, and add them to the corresponding dictionary.
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
        Construct the Input example, which contains the following keys
        text_a (str, required): The content text
        text_b (str, optional): Usually the
        label (str, required): the labels
        guid (str, required): A unique id to one InputExample element
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
    # Map the actual token (in original file) to the actual meaning of it
    VERBALIZER = {"0": ["中立"],
                    "1": ["利好"],
                    "2": ["利空"]}

    @staticmethod
    def available_patterns():
        # Return ids of all available patterns
        return [0]

    @property
    def is_multi_token(self):
        # If the label can contain more than 1 token, return True
        return True

    def get_parts(self, example: InputExample) -> FilledPattern:
        # Organize the elements in InputExample into a designed pattern
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
More information about why and how to create our own dataset can be viewed [here](APPENDIX_TASK.md).

