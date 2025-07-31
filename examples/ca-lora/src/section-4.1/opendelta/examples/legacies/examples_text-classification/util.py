from datasets import load_dataset, load_metric
import torch
import logging


logger = logging.getLogger(__name__)


class DataLoader:
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq"]
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2"]

    def __init__(self, raw_datasets, data_args, model_args, training_args):
        self.raw_datasets = raw_datasets
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.training_args.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        if indices is None:
           indices = self.shuffled_indices(dataset)
        return dataset.select(indices)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def get(self, split):
        if self.data_args.task_name == 'mnli':
            if split == 'validation':
                split = 'validation_mismatched'
            elif split == 'test':
                split = 'validation_matched'
            return self.raw_datasets[split]
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if self.data_args.task_name in self.small_datasets_without_all_splits \
                and split != "train":
            logger.info("Split validation set into test and validation set.")
            dataset = self.raw_datasets['validation']
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif self.data_args.task_name in self.large_data_without_all_splits \
                and split != "test":
            logger.info("Split training set into train and validation set, use validation set as test set.")
            dataset = self.raw_datasets['train']
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, indices)
        elif split == 'train':
            dataset = self.raw_datasets[split]
        else:
            assert split == 'test', print("expected test, but got {}".format(split))
            dataset = self.raw_datasets[split]
        return dataset