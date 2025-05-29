"""Finetuning the library models for sequence classification on GLUE."""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import logging
import json

from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoConfig, AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.label_search import find_labels
from src.dataset import FewShotDataset
from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from src.trainer import Trainer
from src.processors import output_modes_mapping, num_labels_mapping

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


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
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

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    use_space_word: bool = field(
        default=True,
        metadata={"help": "Use space words (e.g., Gpositive) instead of original words."}
    )

    use_seed_labels: bool = field(
        default=False,
        metadata={"help": "Regularize using seed labels"},
    )

    k_likely: int = field(
        default=100,
        metadata={"help": "Take the top-k most (conditionally) likely labels per class."}
    )

    k_neighbors: int = field(
        default=50,
        metadata={"help": "Re-rank by nearest neighbor, and take the top k."}
    )

    n_pairs: int = field(
        default=32,
        metadata={"help": "Number of label pairings to use."}
    )

    output_file: str = field(
        default="out",
        metadata={"help": "Output file"}
    )

    append_output_file: bool = field(
        default=False,
    )

    write_template: bool = field(
        default=False,
    )


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Fix prompt to be true.
    data_args.prompt = True
    data_args.num_sample = 1
    data_args.template_list = None
    data_args.gpt3_in_context_head = False
    data_args.gpt3_in_context_tail = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

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

    if config.model_type == 'roberta':
        model_fn = RobertaForPromptFinetuning
    elif config.model_type == 'bert':
        model_fn = BertForPromptFinetuning
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    set_seed(training_args.seed)

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    # Pass dataset and argument information to the model
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer
    model.return_full_softmax = True

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
    )

    # First we compute zero-shot logits on all of the examples.
    dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=False)

    # Predict logits.
    dataloader = trainer.get_eval_dataloader(dataset)
    output = trainer.prediction_loop(dataloader, description="Evaluation")
    logits = output.predictions[0] if isinstance(output.predictions, (list, tuple)) else output.predictions
    labels = output.label_ids

    # Assign words to labels.
    if data_args.use_seed_labels:
        if data_args.use_space_word:
            seed_labels = {k: "Ġ" + v for k, v in eval(data_args.mapping).items()}
        else:
            seed_labels = eval(data_args.word_mapping)
        seed_labels = [seed_labels[label] for label in dataset.get_labels()]
    else:
        seed_labels = None

    vocab = list(tokenizer.get_vocab())

    # Find best labels.
    label_pairings = find_labels(
        model=trainer.model,
        train_logits=logits,
        train_labels=labels,
        seed_labels=seed_labels,
        k_likely=data_args.k_likely,
        k_neighbors=data_args.k_neighbors,
        top_n=data_args.n_pairs,
        vocab=vocab,
        is_regression=config.num_labels == 1)

    labels = dataset.get_labels()
    if config.num_labels == 1:
        labels = ['0', '1']

    os.makedirs(os.path.dirname(data_args.output_file), exist_ok=True)
    if data_args.append_output_file:
        mode = "a"
    else:
        mode = "w"

    # Write to output.
    with open(data_args.output_file, mode) as f:
        for pairing in label_pairings:
            words = [vocab[i][len("Ġ"):] for i in pairing]
            mapping = {labels[i]: words[i] for i in range(len(labels))}
            if data_args.write_template:
                f.write(data_args.template + "\t")
            f.write(json.dumps(mapping) + "\n")


if __name__ == "__main__":
    main()
