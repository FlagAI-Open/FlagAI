"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union, List
import torch

import numpy as np

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers import AdapterType, AdapterConfig

from src.dataset import FewShotDataset
from src.models import AutoRobertaForMaskedLM
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    # Length of prompt
    prompt_num: int = field(
        default=10,
        metadata={"help": "The length of the prompt"}
    )

    use_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to add the adapter layers"}
    )

@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
 
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    training_params: Optional[List[str]] = field(
        default='all',
        metadata={"help": "The parameters to be trained (all, prompt, adapter or bias)"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.prompt = True
    data_args.prompt_num = model_args.prompt_num

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id] 
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Avoid space in name
    training_args.output_dir = training_args.output_dir.replace(' ', '-')

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    config.prompt_num = model_args.prompt_num

    if config.model_type == 'roberta':
        model_fn = AutoRobertaForMaskedLM
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    # Get our special datasets.
    logger.info("*** Here!!!!! ***")
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=False)
    )
    logger.info("*** Here!!!!! ***")
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=False)
        if training_args.do_eval
        else None
    )
    logger.info("*** Here!!!!! ***")
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=False)
        if training_args.do_predict
        else None
    )


    set_seed(training_args.seed)

    model = model_fn(use_prompt=True,
                     model_name_or_path=model_args.model_name_or_path, config=config)

    if model_args.use_adapter:
        config = AdapterConfig.load('houlsby', reduction_factor=16)
        model.add_adapter('PBAdapter', AdapterType('text_task'), config=config)

    # Pass dataset and argument information to the model
    model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)
            
            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer

    def save_trained_param(model, dir):
        param = {}
        for p in model.named_parameters():
            if p[1].requires_grad:
                param[p[0]] = p[1]
        torch.save(param, dir)
    
    def load_trained_param(model, dir):
        param = torch.load(dir)
        model.load_state_dict(param, strict=False)

    if training_args.do_predict:
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            mnli_mm_data_args.prompt_num = data_args.prompt_num
            mnli_mm_data_args.prompt = True
            test_datasets.append(
                FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", use_demo=False)
            )

    if training_args.do_eval:
        eval_datasets = [eval_dataset]

    final_result = {}
    # Training
    if training_args.do_train:
        train_params_list = training_args.training_params
        logger.info(str(train_params_list))
        if isinstance(train_params_list, str):
            train_params_list = [train_params_list]
        for params in train_params_list:
            model.freeze_model()
            if 'adapter' in params:
                model.train_adapter('PBAdapter')
            if 'prompt' in params:
                model.train_prompt()
            if 'bias' in params:
                model.train_bias()
            if 'all' in params:
                model.train()
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=build_compute_metrics_fn(data_args.task_name)
            )
            result = trainer.train(model_path=model_args.model_name_or_path 
                                   if os.path.isdir(model_args.model_name_or_path) else None) [-1]
            torch.save(result, os.path.join(training_args.output_dir, "result_"+params+".pt"))

            # Use the early stop, so do not save the model in the end (unless specify save_at_last)
            save_trained_param(model, os.path.join(training_args.output_dir, 
                                                   "in_step_model_"+params+'.bin'))
 
            # Save Logs
            torch.save(trainer.state.log_history, os.path.join(training_args.output_dir, "log_history_"+params+'.bin'))

            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)
                torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
                torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        
            # Reload the best checkpoint (for eval)
            model.load_state_dict(torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin")))
            model = model.to(training_args.device)
            trainer.model = model
            if data_args.prompt:
                model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
            if output_modes_mapping[data_args.task_name] == 'regression':
                # lower / upper bounds
                model.lb, model.ub = bound_mapping[data_args.task_name]
            model.model_args = model_args
            model.data_args = data_args
            model.tokenizer = tokenizer

            # #Train Eval
            # train_results = {}
            # logger.info("*** Stage Train Validate ***")

            # trainer.compute_metrics = build_compute_metrics_fn(train_dataset.args.task_name)
            # output = trainer.evaluate(eval_dataset=train_dataset)
            # train_result = output.metrics 

            # output_train_file = os.path.join(
            #     training_args.output_dir, f"train_results_{train_dataset.args.task_name}.txt"
            # )
            # if trainer.is_world_master():
            #     with open(output_train_file, "w") as writer:
            #         logger.info("***** Train results {} *****".format(train_dataset.args.task_name))
            #         for key, value in train_result.items():
            #             logger.info("  %s = %s", key, value)
            #             writer.write("%s = %s\n" % (key, value))
            #             final_result[train_dataset.args.task_name + '_train_' + params + '_' + key] = value
            # train_results.update(train_result)

            # #Evaluation
            # eval_results = {}
            # if training_args.do_eval:
            #     logger.info("*** Stage Validate ***")

            #     for eval_dataset in eval_datasets:
            #         trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            #         output = trainer.evaluate(eval_dataset=eval_dataset)
            #         eval_result = output.metrics 

            #         output_eval_file = os.path.join(
            #             training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            #         )
            #         if trainer.is_world_master():
            #             with open(output_eval_file, "w") as writer:
            #                 logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            #                 for key, value in eval_result.items():
            #                     logger.info("  %s = %s", key, value)
            #                     writer.write("%s = %s\n" % (key, value))
            #                     final_result[eval_dataset.args.task_name + '_dev_' + params + '_' + key] = value
            #         eval_results.update(eval_result)

            # test_results = {}
            # if training_args.do_predict:
            #     logging.info("*** Test ***")

            #     for test_dataset in test_datasets:
            #         trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            #         output = trainer.evaluate(eval_dataset=test_dataset)
            #         test_result = output.metrics

            #         output_test_file = os.path.join(
            #             training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            #         )
            #         if trainer.is_world_master():
            #             with open(output_test_file, "w") as writer:
            #                 logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
            #                 for key, value in test_result.items():
            #                     logger.info("  %s = %s", key, value)
            #                     writer.write("%s = %s\n" % (key, value))
            #                     final_result[test_dataset.args.task_name + '_test_' + params + '_' + key] = value

            #             if training_args.save_logit:
            #                 predictions = output.predictions
            #                 num_logits = predictions.shape[-1]
            #                 logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
            #                 np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            #        test_results.update(test_result)

            load_trained_param(model, os.path.join(training_args.output_dir, "in_step_model_"+params+'.bin'))
            model = model.to(training_args.device)
            trainer.model = model
            if data_args.prompt:
                model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
            if output_modes_mapping[data_args.task_name] == 'regression':
                # lower / upper bounds
                model.lb, model.ub = bound_mapping[data_args.task_name]
            model.model_args = model_args
            model.data_args = data_args
            model.tokenizer = tokenizer

        # Reload the best checkpoint (for eval)
        model.load_state_dict(torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin")))
        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == 'regression':
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer


    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )

 
    #Train Eval
    train_results = {}
    logger.info("*** Train Validate ***")

    trainer.compute_metrics = build_compute_metrics_fn(train_dataset.args.task_name)
    output = trainer.evaluate(eval_dataset=train_dataset)
    train_result = output.metrics 

    output_train_file = os.path.join(
        training_args.output_dir, f"train_results_{train_dataset.args.task_name}.txt"
    )
    if trainer.is_world_master():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results {} *****".format(train_dataset.args.task_name))
            for key, value in train_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
                final_result[train_dataset.args.task_name + '_train_' + key] = value
    train_results.update(train_result)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics 

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)

    with FileLock('log.lock'):
        with open('log', 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if 'evaluation_strategy' in final_result:
                final_result.pop('evaluation_strategy')
            f.write(str(final_result) + '\n')
    
    try:
        os.remove(os.path.join(training_args.output_dir, "pytorch_model.bin"))
    except:
        logger.info("The pytorch_model.bin file doesn't EXIST")
    return eval_results

if __name__ == "__main__":
    main()
