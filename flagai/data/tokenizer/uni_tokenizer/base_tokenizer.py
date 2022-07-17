import logging
logger = logging.getLogger(__name__)
import os
from flagai.model.file_utils import _get_model_files, _get_model_id, _get_vocab_path
from flagai.data.tokenizer.uni_tokenizer.properties import VOCAB_FILE, MERGES_FILE, SP_MODEL_FILE, VOCAB_JSON_FILE
import warnings
from enum import Enum


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class BaseTokenizer(object):
    @classmethod
    def from_pretrained(cls,
                        tokenizer_model_name,
                        cache_dir=None, *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.

        Args:
            tokenizer_model_name (`str`):
                Name of the model associated with the tokenizer
            cache_dir (`str`):
                The directory that contains the vocab files, or will receive the downloaded vocab files
        """
        if cache_dir is None:
            # cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs')
            cache_dir = "/root/.cache/FlagAI/"+tokenizer_model_name
        tokenizer_class = ""
        # search the cache directory for certain files

        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            if SP_MODEL_FILE in files:
                tokenizer_class = "sp"
            elif VOCAB_JSON_FILE in files and MERGES_FILE in files:
                tokenizer_class = "bpe"
            elif VOCAB_FILE in files:
                tokenizer_class = "wp"
        if tokenizer_class == "":
            print("downloading model %s from ModelHub"%tokenizer_model_name)
            files = _get_model_files(tokenizer_model_name)
            model_id = _get_model_id(tokenizer_model_name)
            if SP_MODEL_FILE in files:
                tokenizer_class = "sp"
                _get_vocab_path(cache_dir + '/', SP_MODEL_FILE, model_id, rank=0)
            elif VOCAB_JSON_FILE in files and MERGES_FILE in files:
                tokenizer_class = "bpe"
                _get_vocab_path(cache_dir + '/', VOCAB_JSON_FILE, model_id, rank=0)
                _get_vocab_path(cache_dir + '/', MERGES_FILE, model_id, rank=0)
            elif VOCAB_FILE in files:
                tokenizer_class = "wp"
                _get_vocab_path(cache_dir + '/', VOCAB_FILE, model_id, rank=0)
            else:
                raise FileNotFoundError("Error: no tokenizer files")
        # print(tokenizer_class,22222)
        resolved_vocab_json_file = os.path.join(cache_dir, VOCAB_JSON_FILE)
        resolved_vocab_file = os.path.join(cache_dir, VOCAB_FILE)
        resolved_merges_file = os.path.join(cache_dir, MERGES_FILE)
        resolved_sp_file = os.path.join(cache_dir, SP_MODEL_FILE)
        if tokenizer_class == "wp":
            return cls(vocab_file=resolved_vocab_file, tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name, cache_dir=cache_dir, *inputs, **kwargs)
        elif tokenizer_class == "bpe":
            return cls(vocab_file=resolved_vocab_json_file, merges_file=resolved_merges_file, tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name, cache_dir=cache_dir, *inputs, **kwargs)
        elif tokenizer_class == "sp":
            return cls(sp_model_file=resolved_sp_file, tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name, cache_dir=cache_dir, *inputs, **kwargs)
        else:
            raise NotImplementedError("Cannot find a tokenizer class that matches the files settings in the directory or ModelHub")


    def __init__(self,
                 vocab_file=None,
                 merges_file=None,
                 sp_model_file=None,
                 tokenizer_class=None,
                 tokenizer_model_name=None,
                 cache_dir=None,
                 *inputs,
                 **kwargs):

        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.sp_model_file = sp_model_file
        self.tokenizer_class = tokenizer_class
        self.tokenizer_model_name = tokenizer_model_name
        self.cache_dir = cache_dir
        self.deprecation_warnings = (
            {}
        )
        # self.max_len = int(1e12)

    # def encode_plus(
    #     self,
    #     text,
    #     truncation = False,
    #     max_length = None,
    # ):
    #     """
    #     Tokenize and prepare for the model a sequence or a pair of sequences.
    #
    #     <Tip warning={true}>
    #
    #     This method is deprecated, `__call__` should be used instead.
    #
    #     </Tip>
    #
    #     Args:
    #         text (`str`, `List[str]` or `List[int]` (the latter only for not-fast tokenizers)):
    #             The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
    #             `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
    #             method).
    #         text_pair (`str`, `List[str]` or `List[int]`, *optional*):
    #             Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
    #             the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
    #             method).
    #     """
    #
    #     # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
    #     padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
    #         truncation=truncation,
    #         max_length=max_length,
    #     )
    #
    #     return self._encode_plus(
    #         text=text,
    #         padding_strategy=padding_strategy,
    #         truncation_strategy=truncation_strategy,
    #         max_length=max_length,
    #         **kwargs,
    #     )
    # def _get_padding_truncation_strategies(
    #     self, padding=False, truncation=False, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    # ):
    #     """
    #     Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
    #     and pad_to_max_length) and behaviors.
    #     """
    #     old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
    #     old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)
    #
    #     # Backward compatibility for previous behavior, maybe we should deprecate it:
    #     # If you only set max_length, it activates truncation for max_length
    #     if max_length is not None and padding is False and truncation is False:
    #         if verbose:
    #             if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
    #                 logger.warning(
    #                     "Truncation was not explicitly activated but `max_length` is provided a specific value, "
    #                     "please use `truncation=True` to explicitly truncate examples to max length. "
    #                     "Defaulting to 'longest_first' truncation strategy. "
    #                     "If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy "
    #                     "more precisely by providing a specific strategy to `truncation`."
    #                 )
    #             self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
    #         truncation = "longest_first"
    #     elif truncation is not False:
    #         if truncation is True:
    #             truncation_strategy = (
    #                 TruncationStrategy.LONGEST_FIRST
    #             )  # Default to truncate the longest sequences in pairs of inputs
    #         elif not isinstance(truncation, TruncationStrategy):
    #             truncation_strategy = TruncationStrategy(truncation)
    #         elif isinstance(truncation, TruncationStrategy):
    #             truncation_strategy = truncation
    #     else:
    #         truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
    #
    #     # Set max length if needed
    #     if max_length is None:
    #
    #         if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
    #             if self.model_max_length > int(1e20):  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER:
    #                 if verbose:
    #                     if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
    #                         logger.warning(
    #                             "Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. "
    #                             "Default to no truncation."
    #                         )
    #                     self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
    #                 truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
    #             else:
    #                 max_length = self.model_max_length
    #
    #     return truncation_strategy, max_length, kwargs



if __name__ == '__main__':
    tokenizer = BaseTokenizer.from_pretrained('GLM-large-en')
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("fried chicken makes me happy")))