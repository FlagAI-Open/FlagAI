from collections import OrderedDict
import collections
import abc
import functools
from selectors import EpollSelector
from typing import Callable, List, Mapping
from .utils import pad_punctuation
from examples_prompt.metrics import metrics
from .utils import round_stsb_target
import datasets
import logging
import numpy as np
import torch
import re
import itertools
import os

logger = logging.getLogger(__name__)


from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from typing import List, Dict
from collections import defaultdict
import warnings


from .processor import AbstractTask

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix)


##GLUE
class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    templates_text = {"0": """sentence: {"meta": 'sentence', "shortenable":True} Are there any error in the sentence? {"mask"}""",
    }

    verbalizers = {
        "0":{ "0": "yes", "1": "no"}
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.cola")[split]
        else:
            return datasets.load_dataset('glue', 'cola',
                                     split=split, script_version="master")


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = {
        "0":{"0":"negative","1":"positive"},
        "blenderbot":{"0":"negative","1":"positive"}

    }

    templates_text = {
        "0":"""The sentiment of sentence: "{"meta":"sentence", "shortenable":True}" is {"mask"}.""",
        "blenderbot": """{"meta":"sentence", "shortenable":True} what is the sentiment?"""
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.sst2")[split]
        else:
            return datasets.load_dataset('glue', 'sst2',
                                        split=split, script_version="master")



class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


    templates_text = {
        "0": """sentence1: {"meta": 'sentence1', "shortenable":True}. sentence2: {"meta":"sentence2", "shortenable":True}. Are sentence1 and sentence2 equivalent? {"mask"}.""",
    }

    verbalizers = {
        "0":{"0": "no","1": "yes"}
    }


    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.mrpc")[split]
        else:
            return datasets.load_dataset('glue', 'mrpc', split=split, script_version="master")



class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    templates_text = {"0":
        """question1: {"meta": 'question1', "shortenable":True}. question2: {"meta": 'question2', "shortenable":True} Are question1 and question2 equivalent? {"mask"}."""
    }

    verbalizers = {
        "0":{"0": "no","1": "yes"}
    }


    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.qqp")[split]
        else:
            return datasets.load_dataset('glue', 'qqp',
                                     split=split, script_version="master")



class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


    verbalizers = {
        ""
    }

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]


    templates_text = {
        "0":"""premise: {"meta": 'premise', "shortenable":True}. hypothesis: {"meta": 'hypothesis', "shortenable":True} Does the premise entails the hypothesis? {"mask"}.""",
    }

    verbalizers = {
        "0":{
            "0": "yes",
            "1": "neutral",
            "2": "no",
        }
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.mnli")[split]
        else:
            return datasets.load_dataset('glue', 'mnli', split=split, script_version="master")

    # def preprocessor(self, example, add_prefix=True):
    #     src_texts = ["premise:", example['premise'],
    #                  "hypothesis", example["hypothesis"]]
    #     tgt_texts = [str(example['label'])]
    #     return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    templates_text = {
        "0": """premise: {"meta": 'sentence', "shortenable":True}. hypothesis: {"meta": 'question', "shortenable":True}"""+
        """Does the premise entails the hypothesis? {"mask"}.""",
    }

    verbalizers = {
        "0":{
            "0": "yes",
            "1": "no",
        }
    }


    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.qnli")[split]
        else:
            return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    # def load_dataset(self, split):
    #     return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    # def preprocessor(self, example, add_prefix=True):
    #     src_texts = ["question:", example['question'],
    #                  "sentence:", example["sentence"]]
    #     tgt_texts = [str(example['label'])]
    #     return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

#Tested
class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


    templates_text = {
        "0": """sentence1: {"meta": 'sentence1', "shortenable":True} sentence2: {"meta":"sentence2", "shortenable":True} The answer was {"mask"}.""",
    }

    verbalizers = {
        "0":{"0": "yes",
            "1": "no"
        }
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.rte")[split]
        else:
            return datasets.load_dataset('glue', 'rte',
                                     split=split, script_version="master")



class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = {
        "0":{"0": "True",
            "1": "False",
            }
    }
    templates_text = {"0": """{"meta": 'sentence1',"shortenable":True} Does it mean the following: "{"meta":'sentence2'}"? {"mask"}."""
    }


    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.wnli")[split]
        else:
            return datasets.load_dataset('glue', 'wnli', split=split, script_version="master")


#SuperGLUE
class SuperGLUEBoolQ(AbstractTask):
    name="superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = {
        "0": {
            "0": "no",
            "1": "yes"
        },
    }

    templates_text = {
        "0": """hypothesis: {"meta": "question", "shortenable":True} premise: {"meta":"passage", "shortenable":True} The answer was {"mask"}."""
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.boolq")[split]
        else:
            return datasets.load_dataset('super_glue', 'boolq', split=split, script_version="master")


#
class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    verbalizers = {
        "0":{"0": "yes",
            "1": "no",
            "2": "maybe"
        }
    }
    templates_text = {
        "0": """hypothesis: {"meta": 'hypothesis',"shortenable":True} premise: {"meta":'premise', "shortenable":True} The answer was {"mask"}."""
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.cb")[split]
        else:
            return datasets.load_dataset('super_glue', 'cb', split=split, script_version="master")


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    verbalizers = {
        "0":{
            "0": "1",
            "1": "2",
        }
    }
    templates_text = {
       "0": """choice1: {"meta":"choice1"} choice2: {"meta":"choice2"} premise: {"meta":"premise", "shortenable":True} The {"meta":"question"} answer was choice{"mask"}."""
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.copa")[split]
        else:
            return datasets.load_dataset('super_glue', 'copa', split=split, script_version="master")


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.f1_score,
              metrics.accuracy]
    metric_names = ["f1", "em"]


    verbalizers = {
        "0": {
        "0": "no",
        "1": "yes",
        }
    }
    templates_text = {
        "0": """question: {"meta":"question", "shortenable":False} answer: {"meta":"answer", "shortenable":False, "post_processing": lambda x:x+"."} paragraph: {"meta":"paragraph", "shortenable":True} The answer was {"mask"}."""
    }


    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.multirc")[split]
        else:
            return datasets.load_dataset('super_glue', 'multirc', split=split, script_version="master")

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example):
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        example["question"] = self.remove_markup(example["question"])
        example["answer"] = self.remove_markup(example["answer"])
        example["paragraph"] = self.remove_markup(example["paragraph"])
        return example



class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    verbalizers = {
        "0": {
        "0": "No",
        "1": "Yes",
        }
    }

    templates_text = {
        "0": """sentence1: {"meta":"sentence1"} sentence2: {"meta":"sentence2", "shortenable": True} word: {"meta":"word"} {"mask"}."""
    }

    def load_dataset(self, split):
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.wic")[split]
        else:
            return datasets.load_dataset('super_glue', 'wic', split=split, script_version="master")


# class SuperGLUERecord(AbstractTask):
#     """Convert ReCoRD examples to text2text examples.
#     ReCoRD contains a passage, query containing a '@placeholder' string, and a set
#     of entities that are the possible values of the placeholder. Each train and
#     validation example will have a list of answers, any of which would be
#     considered correct.
#     For example, a typical example from ReCoRD might look like
#     {
#       'passsage': 'This is the passage.',
#       'query': 'A @placeholder is a bird.',
#       'entities': ['penguin', 'potato', 'pigeon'],
#       'answers': ['penguin', 'pigeon'],
#     }
#     which this preprocessor would turn into the following two examples:
#     {
#       'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
#                 'potato, pigeon passage: This is the passage.',
#       'targets': 'penguin',
#     }
#     and
#     {
#       'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
#                 'potato, pigeon passage: This is the passage.',
#       'targets': 'pigeon',
#     }
#     """
#     name = "superglue-record"
#     split_to_data_split = {"train": "train",
#                            "validation": "validation",
#                            "test": "validation"}
#     metric = [metrics.squad]
#     metric_names = ["squad"]

#     def load_dataset(self, split):
#         return datasets.load_dataset('super_glue', 'record', split=split, script_version="master")

#     def preprocessor(self, batch, add_prefix=True):
#         new_batch = collections.defaultdict(list)
#         keys = batch.keys()
#         for values in zip(*batch.values()):
#             ex = {k: v for k, v in zip(keys, values)}
#             # updates the passage.
#             passage = ex['passage']
#             passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
#             passage = re.sub(r'\n@highlight\n', '. ', passage)
#             inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
#             if add_prefix:
#                 inputs = self.name + " " + inputs
#             # duplicates the samples based on  number of answers.
#             num_answers = len(ex["answers"])
#             num_duplicates = np.maximum(1, num_answers)
#             new_batch["source"].extend([inputs] * num_duplicates)
#             new_batch["target"].extend(ex["answers"] if num_answers > 0 else ["<unk>"])
#             new_batch["task"].extend([self.name] * num_duplicates)
#             new_batch["extra_fields"].extend([{"answers": ex["answers"]}]*num_duplicates)
#         return new_batch

#     def map_dataset(self, dataset, add_prefix=True):
#         return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
#             batched=True, remove_columns=dataset.column_names)

class Beans(AbstractTask):
    name = "beans"
    labels_list = ['angular_leaf_spot', 'bean_rust', "healthy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    verbalizers = {
        "clip": {
        "angular_leaf_spot": "angular_leaf_spot",
        "bean_rust": "bean_rust",
        "healthy": "healthy",
        }
    }

    templates_text = {
        "clip":"""a photo of {"mask"} leaf."""
    }

    def load_dataset(self, split):
        # from IPython import embed; embed(header="beans")
        offline = os.environ.get("HF_DATASETS_OFFLINE", "0")
        if offline == '1':
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/beans")[split]
        else:
            return datasets.load_dataset('beans', split=split, script_version="master")

class Wikitext(AbstractTask):
    #wikitext-2-v1
    name = "wikitext"
    # labels_list = ['angular_leaf_spot', 'bean_rust', "healthy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.perplexity]
    metric_names = ["perplexity"]

    verbalizers = {
        "0": {
        }
    }

    templates_text = {
        "0": """{"meta":"text"}"""
    }
    split_valid_to_make_test = True
    def load_dataset(self, split):
        # from IPython import embed; embed(header="beans")
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/wikitext")[split]
        else:
            return datasets.load_dataset('wikitext','wikitext-2-v1', split=split, script_version="master")

class Cifar10(AbstractTask):
    name = "cifar10"

    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            d = datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/cifar10")[split].select(range(100))
            print(d)
            return d
        else:
            return datasets.load_dataset('cifar10', split=split, script_version="master")
    # def preprocessor(self, example):
    #     example_ = {}
    #     example_["image"] = example["image"]
    #     example_["labels"] = example["label"]

    #     return example_
class Fashion_MNIST(AbstractTask):
    name = "Fashion-MNIST"

    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            d = datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/fashion_mnist")[split]
            print(d)
            return d
        else:
            return datasets.load_dataset('fashion_mnist', split=split, script_version="master")

TASK_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        # ('superglue-record', SuperGLUERecord)
        ('beans', Beans),
        ('wikitext',Wikitext),
        ('cifar10',Cifar10),
        ('fashion_mnist',Fashion_MNIST)
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, data_args, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, data_args, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
