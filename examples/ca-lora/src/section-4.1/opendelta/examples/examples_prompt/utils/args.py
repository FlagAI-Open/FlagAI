from dataclasses import dataclass, field
from typing import Optional, List
from transformers import HfArgumentParser
from pathlib import Path
import sys



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
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    num_classes:Optional[int]=field(
        default=None, metadata={"help": "The number of classes, used to initialize classification models"}
    )



from transformers import TrainingArguments as HfTrainingArguments
# run_seq2seq parameters.

@dataclass
class TrainingArguments(HfTrainingArguments):
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                                 "the model."})
    do_test: Optional[bool] = field(default=False, metadata={"help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    compute_time: Optional[bool] = field(default=True, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(default=True, metadata={"help": "if set, measures the memory"})
    is_seq2seq: Optional[bool] = field(default=True, metadata={"help": "whether the pipeline is a seq2seq one"})
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    push_to_hf: Optional[bool] = field(default=False, metadata={"help": "Push the model to huggingface model hub."})
    push_to_dc: Optional[bool] = field(default=True, metadata={"help": "Push the model to delta center."})








@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the evaluation dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    test_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the test dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."}
    )
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    datasets_load_from_disk: Optional[bool] = field(
        default=False, metadata={"help": "Whether to load datasets from disk"}
    )
    datasets_saved_path: Optional[str] = field(
        default=None, metadata={"help": "the path of the saved datasets"}
    )
    data_sample_seed: Optional[int] = field(default=42, metadata={"help": "seed used to shuffle the data."})


    model_parallel: Optional[bool] = field(default=False, metadata={"help": "whether apply model parallelization"})

    def __post_init__(self):
        if self.task_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.test_max_target_length is None:
            self.test_max_target_length = self.max_target_length



import dataclasses

@dataclass
class DeltaArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    delta_type: str= field(default="", metadata={"help": "the type of delta"})
    backbone_model: Optional[str] = field(
        default="", metadata={"help": "the backbone model"}
    )
    model_path_public: Optional[str] = field(
        default="", metadata={"help": "the path (url) of the publicly available backbone model"}
    )
    modified_modules: Optional[List[str]] = field(
        default_factory=lambda: None, metadata={"help": "the modules inside the backbone to be modified"}
    )
    unfrozen_modules: Optional[List[str]] = field(
        default_factory=lambda:["deltas"], metadata={"help": "the modules inside the backbone or in the delta modules that need to be unfrozen"}
    )
    finetuned_delta_path: Optional[str] = field(
        default=None, metadata={"help": "the path of the finetuned delta model"}
    )
    force_download: Optional[bool] = field(
        default=False, metadata={"help": "whether to download the checkpoint form delta center no matter whether it exists"}
    )
    local_files_only: Optional[bool] = field(
        default=False, metadata={"help": "whether not to look for file in delta center"}
    )
    delta_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The cache path defined by user. If not set, we will firstly look into the"+
        " working directory and then into the default cache path (ususally ~/.cache/delta_center)."}
    )
    delay_push: Optional[bool] = field(
        default=True, metadata={
            'help':'whether push the checkpoint to delta center later.'
        }
    )

    def merge_arguments(self, objb):
        print(objb)
        self.__class__ = dataclasses.make_dataclass('DeltaArgument', fields=[(s.name, s.type, getattr(objb, s.name)) for s in dataclasses.fields(objb)], bases=(DeltaArguments,))




@dataclass
class AdapterArguments:
    bottleneck_dim: Optional[int] = field(
        default=24, metadata={"help": "the dimension of the bottleneck layer"}
    )
@dataclass
class LoRAArguments:
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "the rank of the LoRA metrics."}
    )
@dataclass
class PrefixArguments:
    pass
@dataclass
class BitFitArguments:
    pass
@dataclass
class SoftPromptArguments:
    soft_token_num: Optional[int] = field(
        default=100, metadata={"help": "the num of soft tokens."}
    )

@dataclass
class CompacterArguments:
    pass
@dataclass
class LowRankAdapterArguments:
    pass

# from opendelta.delta_models.adapter import AdapterConfig
# from opendelta.delta_models.bitfit import BitFitConfig
# from opendelta.delta_models.compacter import CompacterConfig
# from opendelta.delta_models.lora import LoraArguments
# from opendelta.delta_models.low_rank_adapter import LowRankAdapterConfig
# from opendelta.delta_models.prefix import PrefixConfig
# from opendelta.delta_models.soft_prompt import SoftPromptConfig
# DELTAARGMAP = {
#     "adapter": AdapterConfig,
#     "lora":LoraArguments,
#     "prefix":PrefixConfig,
#     "bitfit":BitFitConfig,
#     "soft_prompt":SoftPromptConfig,
#     "compacter":CompacterConfig,
#     "low_rank_adapter":LowRankAdapterConfig

# }

DELTAARGMAP = {
    "adapter": AdapterArguments,
    "lora":LoRAArguments,
    "prefix":PrefixArguments,
    "bitfit":BitFitArguments,
    "soft_prompt":SoftPromptArguments,
    "compacter":CompacterArguments,
    "low_rank_adapter":LowRankAdapterArguments

}

# TODO: add more specific delta arguments



class RemainArgHfArgumentParser(HfArgumentParser):
    '''This is a more powerful version of argument parser.
    It can receiven both command line arguments and json file arguments.
    The command line arguments will override the json file arguments.
    The parser will load the specific delta arguments (e.g. Adapter's)
    according to the delta_type argument. And merge the specific delta arguments
    with the common delta arguments.
    '''
    def parse_json_file_with_cmd_args(self, json_file: str, command_line_args=None, return_remaining_args=True ):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """

        import json
        from pathlib import Path



        data = json.loads(Path(json_file).read_text())


        data_str = ""
        if command_line_args is None:
            command_line_args = []
        for key in data:
            if "--"+key not in command_line_args:
                if isinstance(data[key], list):
                    data_str += "--"+key
                    for elem in data[key]:
                        data_str+=" "+ str(elem)
                    data_str += " "
                else:
                    data_str+= "--" + key + " " + str(data[key]) + " "

        data_list = data_str.split()
        data_list += command_line_args


        if return_remaining_args:
            outputs, remain_args = self.parse_args_into_dataclasses(args=data_list, return_remaining_strings=return_remaining_args)
            for d in outputs:
                if isinstance(d, DeltaArguments): # merge the specific delta arguments
                    d.merge_arguments(outputs[-1])

            return  [*(outputs[:-1]), remain_args]
        else:
            outputs = self.parse_args_into_dataclasses(args=data_list, return_remaining_strings=return_remaining_args)
            for d in outputs:
                if isinstance(d, DeltaArguments):
                    d.merge_arguments(outputs[-1])
            return [*(outputs[:-1]),]

    def parse_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None
    ):
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)

        # conditionally add delta arguments
        deltatype_args = DELTAARGMAP[namespace.delta_type]
        self.dataclass_types.append(deltatype_args)
        self._add_dataclass_arguments(deltatype_args)

        # parse the arguments again, this time with the specific delta type's arguments
        namespace, remaining_args = self.parse_known_args(args=args)


        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            return outputs

        # namespace, remaining_args = self.parse_known_args(args=data_list)

        # print("Here", command_line_args, data_list,namespace, remaining_args)
        # data.update(remain_args)

        # outputs = []
        # for dtype in self.dataclass_types:
        #     keys = {f.name for f in dataclasses.fields(dtype) if f.init}
        #     inputs = {k: namespace.get(k) for k in list(data.keys()) if k in keys}
        #     obj = dtype(**inputs)
        #     outputs.append(obj)

        # # remain_args = argparse.ArgumentParser()
        # remain_args.__dict__.update(remain_args)
        # if return_remaining_args:
        #     return (*outputs, remain_args)
        # else:
        #     return (*outputs,)


