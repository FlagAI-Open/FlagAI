# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

from flagai.data.tokenizer import Tokenizer
from flagai.model.glm_model import GLMForSeq2Seq
from flagai.trainer import Trainer
from flagai.data.dataset import ConstructBlockStrategy
from flagai.data.dataset import BlockDataset
from flagai.data.dataset.block.data_utils import split_ds, get_dataset_lazy, add_args
from flagai.data.dataset.superglue.control import DEFAULT_METRICS
from flagai.test_utils import PretrainDatasetArguments

if __name__ == '__main__':

    trainer = Trainer(env_type='bmtrain',
                      epochs=1,
                      batch_size=1,
                      eval_interval=1000,
                      log_interval=10,
                      experiment_name='glm_large',
                      pytorch_device='cuda',
                      load_dir=None,
                      lr=1e-4,
                      num_gpus = 2,
                      weight_decay=1e-2,
                      save_interval=8000,
                      hostfile='./hostfile',
                      deepspeed_config='./deepspeed.json')
    model_name = 'GLM-large-ch'
    tokenizer = Tokenizer.from_pretrained(model_name)
    ds_args = PretrainDatasetArguments()
    ds_args = add_args(ds_args, tokenizer)
    model = GLMForSeq2Seq.from_pretrain(download_path='./checkpoints',model_name=model_name)
    def create_dataset(tokenizer, should_split):
        dataset = get_dataset_lazy("./data",
                                   tokenizer=tokenizer,
                                   pre_tokenize=True,
                                   num_processes=10,
                                   no_lazy_loader=True)
        if should_split:
            datasets = split_ds(dataset, split=[.8, .2, .0], shuffle=True)
        else:
            datasets = [dataset]

        datasets = [
            BlockDataset(ds,
                         tokenizer,
                         max_seq_len=512,
                         sample_across_doc=True,
                         non_sentence_start=0.0) if ds is not None else None
            for ds in datasets
        ]
        return datasets

    datasets = create_dataset(tokenizer, should_split=True)

    collate_fn = None
    if ds_args.block_lm:
        collate_fn = ConstructBlockStrategy(
            tokenizer, 512, eod_token=tokenizer.get_command_id('eos'))
    metric_methods = DEFAULT_METRICS['pretrain']
    trainer.train(model,
                  collate_fn=collate_fn,
                  train_dataset=datasets[0],
                  valid_dataset=datasets[1],
                  metric_methods=metric_methods)
