# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from torch.utils.data import Dataset
from .control import SuperGlueProcessor
from collections import Counter
import numpy as np
import os
# from flagai.data.dataset.superglue.pvp import PVPS
# from flagai.data.dataset.superglue.control import PVPS
print_rank_0 = print
TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
TRUE_DEV_SET = "true_dev"
UNLABELED_SET = "unlabeled"

SPLIT_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, TRUE_DEV_SET, UNLABELED_SET]


class SuperGlueDataset(Dataset):

    def __init__(self,
                 task_name,
                 data_dir,
                 dataset_type,
                 tokenizer,
                 for_train=False,
                 few_superglue=False,
                 cloze_eval=False,
                 multi_token=False):
        """
        Args:
            task_name: (str):
                The name of the task to be processed.
                The value is currently selected from ['boolq','cb','copa','multirc','record','rte','wic','wsc']
                Details for these tasks can be viewed here: https://super.gluebenchmark.com/tasks
            data_dir: (str)
                The root directory that contains the data directories of all non-sequential tasks
            dataset_type (str)
                The type of dataset, selected from SPLIT_TYPES => ["train", "dev", "test", "true_dev", "unlabeled"]
            tokenizer: (Tokenizer)
                The selected tokenizer for this task
            for_train: (bool)
                If for_train is True, the dev set will be fed into training process
        """
        try:
            from datasets import load_dataset
        except Exception:
            raise Exception("datasets is required! pip install datasets")

        self.processor = SuperGlueProcessor().get_processor(
            data_dir, task_name)

        self.processor = self.processor(few_superglue=few_superglue)

        print_rank_0(
            f"Creating {task_name} dataset from file at {data_dir} (split={dataset_type})"
        )
        self.dataset_name = f"{task_name}-{dataset_type}"
        self.tokenizer = tokenizer

        if os.path.exists(os.path.join(data_dir, task_name)):
            data_dir = os.path.join(data_dir, task_name)
        else:
            raise NameError("task name not found")

        if dataset_type == DEV_SET:
            example_list = self.processor.get_dev_examples(data_dir,
                                                           for_train=for_train)
        elif dataset_type == TEST_SET:
            example_list = self.processor.get_test_examples(data_dir)
        elif dataset_type == TRUE_DEV_SET:
            example_list = self.processor.get_true_dev_examples(data_dir)
        elif dataset_type == TRAIN_SET:
            if task_name == "wsc":
                example_list = self.processor.get_train_examples(
                    data_dir, cloze_eval=cloze_eval)
            else:
                example_list = self.processor.get_train_examples(data_dir)
        elif dataset_type == UNLABELED_SET:
            example_list = self.processor.get_unlabeled_examples(data_dir)
            for example in example_list:
                example.label = self.processor.get_labels()[0]
        else:
            raise ValueError(
                f"'dataset_type' must be one of {SPLIT_TYPES}, got '{dataset_type}' instead"
            )

        if dataset_type == TEST_SET:
            self.labeled = False
        else:
            self.labeled = True

        label_distribution = Counter(example.label for example in example_list)
        print_rank_0(
            f"Returning {len(example_list)} {dataset_type} examples with label dist.: {list(label_distribution.items())}"
        )

        example_list.sort(key=lambda x: x.num_choices)
        self.example_list = example_list
        self.num_classes = len(label_distribution)
        self.examples = {example.guid: example for example in example_list}

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        sample_idx = idx % len(self.example_list)
        example = self.example_list[sample_idx]
        return example
