# coding=utf-8
# Copyright OpenDelta Team and THUNLP lab. All rights reserved.
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
A unified runing scripts for most models to do down stream tasks in a
prompt learning fashion, i.e., No classification head, all tasks are casted
to mask prediction or span prediction tasks.

Processing relevant to different backbone models are stored in ../backbones/

Adding A few lines to integrate the Delta tuning methods.

You can also adapt this script on your own tasks.
"""

import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.join(os.getcwd(), "../"))
# sys.path.append(os.path.join(os.getcwd(), "/mnt/sfs_turbo/zhangzhen/OpenDelta"))
sys.path.append(os.path.join(os.getcwd()))

import functools
import logging
import torch
import json
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    # HfArgumentParser,
    # MBartTokenizer,
    # default_data_collator,
    Trainer,
    Seq2SeqTrainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint

from data_processors import AutoTask #, #TaskDataCollatorForSeq2Seq, AutoPostProcessor, data_collator
from utils import read_json, save_json
from utils.args import ModelArguments, TrainingArguments, DataTrainingArguments, DeltaArguments, RemainArgHfArgumentParser


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DeltaArguments))

    # You can provide a json file with contains the arguments and use the --argument some_arg to override or append to  the json file.
    json_file, cmd_args = (os.path.abspath(sys.argv[1]), sys.argv[2:]) if sys.argv[1].endswith(".json") else (None, sys.argv[1:])
    model_args, data_args, training_args, delta_args, remain_args = parser.parse_json_file_with_cmd_args(json_file=json_file, command_line_args=cmd_args)
    logger.warning("The following arguments not used! {}".format(remain_args))

    logger.info(f"The results will be used in {training_args.output_dir}/results.json")
    # exit()
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
    # logger.info("Training/evaluation parameters %s", training_args, model_args, data_args, delta_args)
    logger.info("{}\n{}\n{}\n{}".format(training_args, model_args, data_args, delta_args))


    # Set seed before initializing model.
    set_seed(training_args.seed)



    if os.path.basename(model_args.model_name_or_path).startswith("t5") \
        or os.path.basename(model_args.model_name_or_path).startswith("long-t5") :
        from examples_prompt.backbones.t5 import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.t5 import Trainer, DataCollator
    elif  os.path.basename(model_args.model_name_or_path).startswith("blenderbot"):
        from examples_prompt.backbones.blenderbot import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.blenderbot import Trainer, DataCollator
    elif os.path.basename(model_args.model_name_or_path).startswith("roberta") \
        or os.path.basename(model_args.model_name_or_path).startswith("bert") \
          or os.path.basename(model_args.model_name_or_path).startswith("albert") \
            or os.path.basename(model_args.model_name_or_path).startswith("xlm-roberta") \
                or os.path.basename(model_args.model_name_or_path).startswith("deberta") :
        from examples_prompt.backbones.bert import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.bert import Trainer, DataCollator
    elif os.path.basename(model_args.model_name_or_path).startswith("beit"):
        from examples_prompt.backbones.beit import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.beit import Trainer, DataCollator
    elif os.path.basename(model_args.model_name_or_path).startswith("bart"):
        from examples_prompt.backbones.bart import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.bart import Trainer, DataCollator
    elif os.path.basename(model_args.model_name_or_path).startswith("bigbird"):
        from examples_prompt.backbones.bigbird import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.bigbird import Trainer, DataCollator
    elif os.path.basename(model_args.model_name_or_path).startswith("clip"):
        from examples_prompt.backbones.clip import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.clip import Trainer, DataCollator
    elif os.path.basename(model_args.model_name_or_path).startswith("opt") \
        or os.path.basename(model_args.model_name_or_path).startswith("gpt"):
        from examples_prompt.backbones.opt import get_backbone, preprocess_function, mask_token_func, get_remove_columns, get_prompts
        from examples_prompt.backbones.opt import Trainer, DataCollator





    config, tokenizer, model = get_backbone(model_args=model_args)

    # model parallelize
    if hasattr(training_args, "model_parallel") and training_args.model_parallel:
        logger.info('parallelize model!')
        model.parallelize()

    from bigmodelvis import Visualization
    Visualization(model).structure_graph()

    if delta_args.delta_type.lower() != "none":
        from opendelta import AutoDeltaConfig,AutoDeltaModel
        from dataclasses import asdict
        delta_config = AutoDeltaConfig.from_dict(asdict(delta_args))
        delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=model)
        delta_model.freeze_module(set_state_dict = True)
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)





    performance_metrics = {}




    non_empty_splits_names = []
    if training_args.do_train:
        non_empty_splits_names.append("train")
    if training_args.do_eval:
        non_empty_splits_names.append("eval")
    if training_args.do_test:
        non_empty_splits_names.append("test")
    splits = {}
    for split_name in ['train', 'eval', 'test']:
        if split_name not in non_empty_splits_names:
            splits[split_name] = None
            continue

        task = AutoTask.get(data_args.task_name,
                            data_args.dataset_config_name,
                            data_args=data_args,
                            seed=data_args.data_sample_seed)

        dataset =  task.get(split=split_name,
                            split_validation_test=training_args.split_validation_test,
                            n_obs=data_args.max_train_samples)



        template, _verbalizer, tokenizer_wrapper = get_prompts(task, tokenizer, data_args)


        dataset = dataset.map(
                            functools.partial(preprocess_function,
                                            data_args=data_args,
                                            tokenizer=tokenizer,
                                            template=template,
                                            verbalizer=_verbalizer,
                                            tokenizer_wrapper=tokenizer_wrapper,
                                            split=split_name),
                            batched=False,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=get_remove_columns(list(dataset.features.keys())),
                            load_from_cache_file=not data_args.overwrite_cache,
                        )
        # from IPython import embed; embed()
        splits[split_name] = dataset
        if split_name == "eval":
            eval_task = task
            verbalizer = _verbalizer



    trainer = Trainer(
        model=model,
        verbalizer=verbalizer,
        eval_task=eval_task,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits['eval'],
        tokenizer=tokenizer,
        data_collator=DataCollator(tokenizer),
    )


    def save_training_config(config_file, output_dir):
        json_data = read_json(config_file)
        save_json(os.path.join(output_dir, "training_config.json"), json_data)


    # Saves training config.
    if trainer.is_world_process_zero():
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
            data_args.max_train_samples if data_args.max_train_samples is not None else len(splits['train'])
        )
        train_metrics["train_samples"] = min(max_train_samples, len(splits['train']))
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        logger.info("Efficiency Statistics {}".format(performance_metrics))
        trainer.save_metrics("performance", performance_metrics)

    # Evaluation
    all_results = {}

    all_results['evaluate'] = {}

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=splits['eval'],
        )
        trainer.log_metrics(f"{data_args.task_name}_eval", metrics)
        trainer.save_metrics(f"{data_args.task_name}_eval", metrics)
        all_results['evaluate'][data_args.task_name] = metrics

    # Test
    all_results['test'] = {}
    if training_args.do_test:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=splits['test'],
        metric_key_prefix="test"
        )
        trainer.log_metrics(f"{data_args.task_name}_test", metrics)
        trainer.save_metrics(f"{data_args.task_name}_test", metrics)
        all_results['test'][data_args.task_name] = metrics

    # from opendelta.utils.delta_hub import create_hub_repo_name
    # from opendelta.utils.delta_center import create_delta_center_args, create_repo_name

    # repo_name = create_hub_repo_name(root="DeltaHub",
    #                      dataset=data_args.task_name,
    #                      delta_type = delta_args.delta_type,
    #                      model_name_or_path= model_args.model_name_or_path)

    # center_args =
    # repo_name = create_repo_name(prefix="", center_args=center_args)
    # all_results['repo_name'] = repo_name


    delta_model.save_finetuned(finetuned_delta_path=delta_args.finetuned_delta_path,
                               push_to_dc=training_args.push_to_dc,
                               center_args={"test_performance":all_results['test'][data_args.task_name]['test_average_metrics'],
                                            },
                               center_args_pool = {**vars(model_args), **vars(data_args), **vars(training_args), **vars(delta_args)},
                               list_tags = ['NLI'],
                               dict_tags = {'purpose':'for testing'},
                               delay_push=True,
                               test_result=all_results['test']
                            )



    with open(f"{training_args.output_dir}/results.json", 'w') as fout:
        string = json.dumps(all_results, indent=4,sort_keys=True)
        fout.write(string+"\n")

    return all_results




if __name__ == "__main__":
    result = main()

