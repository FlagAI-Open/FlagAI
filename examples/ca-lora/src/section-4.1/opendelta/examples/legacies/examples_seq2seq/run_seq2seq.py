# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import functools
import logging
# from opendelta.utils.delta_center import create_hub_repo_name
import torch
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import sys
import subprocess
from typing import Optional, List

from datasets import load_dataset, load_metric, concatenate_datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
# from ..seq2seq.utils import get_adapter_config
from examples_seq2seq.data_processors import AutoTask, TaskDataCollatorForSeq2Seq, AutoPostProcessor
from examples_seq2seq.seq2seq_trainer import Seq2SeqTrainer
# from training_args import AdapterTrainingArguments
from examples_seq2seq.trainers.trainer_utils import save_training_config
from dataclasses import dataclass, field

from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
from examples_seq2seq.trainers.model_args import ModelArguments
from examples_seq2seq.trainers.trainer_args import TrainingArguments, DataTrainingArguments

import tensorboardX
tb_writer = tensorboardX.SummaryWriter("Delta_Memory")

logger = logging.getLogger(__name__)

def run_command(command):
    output = subprocess.getoutput(command)
    return output


TASK_TO_METRICS = {"mrpc": ["accuracy", "f1"],
                  "cola": ['matthews_correlation'],
                  "stsb": ['pearson', 'spearmanr'],
                  'sst2': ['accuracy'],
                  "mnli": ["accuracy"],
                  "mnli_mismatched": ["accuracy"],
                  "mnli_matched": ["accuracy"],
                  "qnli": ["accuracy"],
                  "rte": ["accuracy"],
                  "wnli": ["accuracy"],
                  "qqp": ["accuracy", "f1"],
                  "superglue-boolq": ["accuracy"],
                  "superglue-rte": ["accuracy"],
                  "superglue-cb": ["f1_multiclass", "accuracy"],
                  "superglue-copa": ["accuracy"],
                  "superglue-multirc": ["f1", "em"],
                  "superglue-wic": ["accuracy"],
                  "superglue-wsc.fixed": ["accuracy"],
                  "superglue-record": ["f1", "em"]
         }


class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True ):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        import argparse
        import json
        from pathlib import Path
        import dataclasses

        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)

# from transformers.trainer_callback import TrainerCallback

# class MyCallback(TrainerCallback):
#     def __init__(self, *args, **kwargs):
#         self.delta_args = kwargs.pop("delta_args")
#         self.trainer_args = kwargs.pop("trainer_args")
#         self.model_args = kwargs.pop("model_args")
#         super(MyCallback, self).__init__(*args, **kwargs)


#     maxcudamem = 0
#     def on_step_end(self, args, state, control, **kwargs ):
#         glb_step = state.global_step
#         cudamem = 0
#         realcudamem =0
#         for device_id in range(torch.cuda.device_count()):
#             cudamem += torch.cuda.memory_allocated(f"cuda:{device_id}")/1024**3
#             realcudamem += torch.cuda.max_memory_allocated(f"cuda:{device_id}")/1024**3
#             torch.cuda.reset_peak_memory_stats(f"cuda:{device_id}")
#         self.maxcudamem = max(self.maxcudamem, realcudamem)
#         self.cudamem = cudamem
#         # self.tb_writer.add_scalar("Static Memory (GB)", cudamem, glb_step)
        # self.tb_writer.add_scalar("Runtime Memory (GB)", realcudamem, glb_step)
        # self.tb_writer.add_scalar("Peak Memory (GB)", self.maxcudamem, glb_step)
        # if glb_step > 50:
        #     content = f"{self.delta_args.delta_type}\t{self.trainer_args.per_device_train_batch_size}\t{self.model_args.model_name_or_path}\t{self.cudamem}\t{self.maxcudamem}\n"
        #     with open("memory_data.txt", 'a') as fout:
        #         fout.write(content)
        #     exit()









def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, delta_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, delta_args = parser.parse_args_into_dataclasses()


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            '''
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            '''
            pass
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))


    if delta_args.delta_type.lower() != "none":
        from opendelta import AutoDeltaConfig,AutoDeltaModel
        delta_config = AutoDeltaConfig.from_dict(vars(delta_args))
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=model)
        delta_model.freeze_module(set_state_dict = True)
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)


    # model parallelize
    # if hasattr(training_args, "model_parallel") and training_args.model_parallel:
    #     logger.info('parallelize model!')
    model.parallelize()

    data_args.dataset_name = [data_args.task_name]
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(data_args.eval_dataset_config_name)
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(data_args.test_dataset_config_name)

    # Temporarily set max_target_length for training.
    #max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples, max_target_length):
        # max_target_length += 1
        # model_inputs = tokenizer([s+"<extra_id_0>" for s in examples['source']], max_length=data_args.max_source_length,
        #                          padding=padding, truncation=True)
        # # Setup the tokenizer for targets
        # with tokenizer.as_target_tokenizer():
        #     labels = tokenizer(['<extra_id_0>'+t for t in examples['target']], max_length=max_target_length, padding=padding, truncation=True)
        model_inputs = tokenizer([s for s in examples['source']], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([t for t in examples['target']], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}
    if training_args.do_train:
        train_datasets = [AutoTask.get(dataset_name,
                                       dataset_config_name,
                                       seed=data_args.data_sample_seed).get(
            split="train",
            split_validation_test=training_args.split_validation_test,
            add_prefix=True,
            n_obs=data_args.max_train_samples)
            for dataset_name, dataset_config_name\
            in zip(data_args.dataset_name, data_args.dataset_config_name)]
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(\
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)\
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function, max_target_length=max_target_lengths[i]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, # if train_dataset != "superglue-record" else column_names+["answers"],
                load_from_cache_file=not data_args.overwrite_cache,
            )
        train_dataset = concatenate_datasets(train_datasets)

    if training_args.do_eval:
        eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
            seed=data_args.data_sample_seed).get(
            split="validation",
            split_validation_test=training_args.split_validation_test,
            add_prefix=True,
            n_obs=data_args.max_val_samples)
            for eval_dataset, eval_dataset_config in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]
        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                    functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names, # if name != "superglue-record" else column_names+["answers"],
                    load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_test:
        test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
            seed=data_args.data_sample_seed).get(
            split="test",
            split_validation_test=training_args.split_validation_test,
            add_prefix=True,
            n_obs=data_args.max_test_samples)
            for test_dataset, test_dataset_config in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}
        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length( \
            tokenizer=tokenizer, default_max_length=data_args.max_target_length) \
            for dataset_name, dataset_config_name in zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]
        for k, name in enumerate(test_datasets):
            test_datasets[name] = test_datasets[name].map(
                    functools.partial(preprocess_function, max_target_length=max_target_lengths[k]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )


    # Metric, we assume we have only one training task.
    eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric\
        for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    # Extracts the extra information needed to evaluate on each dataset.
    # These information are only used in the compute_metrics.
    # We will assume that the test/eval dataloader does not change the order of
    # the data.
    data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
                 "test": test_datasets[data_args.test_dataset_name[0]]['extra_fields'],
                 "train": train_dataset['extra_fields']}
    def compute_metrics(eval_preds):
        preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(preds, labels, data_info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        delta_args=delta_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=list(eval_datasets.values())[0] if training_args.do_eval else None,
        data_info = data_info,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]],
    )

    # trainer.add_callback(MyCallback(trainer_args=training_args, delta_args=delta_args, model_args=model_args))


    # Saves training config.
    if trainer.is_world_process_zero():
       os.makedirs(training_args.output_dir, exist_ok=True)
       save_training_config(sys.argv[1], training_args.output_dir)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if training_args.compute_time:
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end)/(1000*60)
            performance_metrics.update({"total_time in minutes ": total_time})

        trainer.save_model()  # Saves the tokenizer too for easy upload
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        print(performance_metrics)
        trainer.save_metrics("performance", performance_metrics)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset,
               max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
            )
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        results['evaluate'] = metrics

    # Test
    if training_args.do_test:
        logger.info("*** Test ***")
        for task, test_dataset in test_datasets.items():
            metrics = trainer.evaluate(eval_dataset=test_dataset,
              max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
              metric_key_prefix="test"
            )
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
        results['test'] = metrics

    repo_name = create_hub_repo_name(root="DeltaHub",
                         dataset=data_args.task_name,
                         delta_type = delta_args.delta_type,
                         model_name_or_path= model_args.model_name_or_path)
    results['repo_name'] = repo_name
    if training_args.push_to_hub: # TODO add description here
        delta_model.save_finetuned(push_to_hub=True, save_directory=repo_name, use_auth_token=True)
        # trainer.push_to_hub(**kwargs)
    else:
        delta_model.save_finetuned(push_to_hub=False, save_directory=repo_name, use_auth_token=True)

    return results




if __name__ == "__main__":
    result = main()
    import json
    with open("collect_result.jsonl", 'a') as fout:
        string = json.dumps(result, indent=4,sort_keys=True)
        fout.write(string+"\n")
    print(result)
