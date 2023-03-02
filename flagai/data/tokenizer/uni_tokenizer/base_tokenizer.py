import os
from flagai.model.file_utils import _get_model_files, _get_model_id, _get_vocab_path
from flagai.data.tokenizer.uni_tokenizer.properties import VOCAB_FILE, MERGES_FILE, SP_MODEL_FILE, VOCAB_JSON_FILE, TOKENIZER_JSON_FILE, SPECIAL_TOKENS_MAP
import warnings


class BaseTokenizer(object):

    @classmethod
    def from_pretrained(cls,
                        tokenizer_model_name,
                        cache_dir=None,
                        *inputs,
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
            # cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs', f"{tokenizer_model_name}")
            cache_dir = './checkpoints/' + tokenizer_model_name
        tokenizer_class = ""
        # search the cache directory for certain files
        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            if SP_MODEL_FILE in files:
                tokenizer_class = "sp"
            elif MERGES_FILE in files:
                tokenizer_class = "bpe"
            elif VOCAB_FILE in files:
                tokenizer_class = "wp"
        if tokenizer_class == "":
            print("downloading model %s from ModelHub" % tokenizer_model_name)
            files = _get_model_files(tokenizer_model_name)
            model_id = _get_model_id(tokenizer_model_name)
            if SP_MODEL_FILE in files:
                tokenizer_class = "sp"
                _get_vocab_path(cache_dir + '/',
                                SP_MODEL_FILE,
                                model_id,
                                rank=0)
            elif MERGES_FILE in files:
                tokenizer_class = "bpe"
                _get_vocab_path(cache_dir + '/', MERGES_FILE, model_id, rank=0)
                if VOCAB_JSON_FILE in files:
                    _get_vocab_path(cache_dir + '/',
                                    VOCAB_JSON_FILE,
                                    model_id,
                                    rank=0)
            elif VOCAB_FILE in files:
                tokenizer_class = "wp"
                _get_vocab_path(cache_dir + '/', VOCAB_FILE, model_id, rank=0)
            else:
                raise FileNotFoundError("Error: no tokenizer files")
        resolved_vocab_json_file = os.path.join(
            cache_dir, VOCAB_JSON_FILE) if VOCAB_JSON_FILE in files else None
        resolved_vocab_file = os.path.join(cache_dir, VOCAB_FILE)
        resolved_merges_file = os.path.join(cache_dir, MERGES_FILE)
        resolved_sp_file = os.path.join(cache_dir, SP_MODEL_FILE)
        special_tokens_map = os.path.join(cache_dir, SPECIAL_TOKENS_MAP)
        resolved_tokenizer_json_file = os.path.join(cache_dir, TOKENIZER_JSON_FILE)
        if tokenizer_class == "wp":
            return cls(vocab_file=resolved_vocab_file,
                       tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name,
                       special_tokens_map=special_tokens_map,
                       cache_dir=cache_dir,
                       *inputs,
                       **kwargs)
        elif tokenizer_class == "bpe":
            return cls(vocab_file=resolved_vocab_json_file,
                       merges_file=resolved_merges_file,
                       tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name,
                       special_tokens_map=special_tokens_map,
                       cache_dir=cache_dir,
                       *inputs,
                       **kwargs)
        elif tokenizer_class == "sp":
            return cls(sp_model_file=resolved_sp_file,
                       tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name,
                       tokenizer_json_file=resolved_tokenizer_json_file,
                       special_tokens_map=special_tokens_map,
                       cache_dir=cache_dir,
                       *inputs,
                       **kwargs)
        else:
            raise NotImplementedError(
                "Cannot find a tokenizer class that matches the files settings in the directory or ModelHub"
            )

    def __init__(self,
                 vocab_file=None,
                 merges_file=None,
                 sp_model_file=None,
                 tokenizer_json_file=None,
                 tokenizer_class=None,
                 tokenizer_model_name=None,
                 special_tokens_map=None,
                 cache_dir=None,
                 *inputs,
                 **kwargs):

        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.sp_model_file = sp_model_file
        self.tokenizer_class = tokenizer_class
        self.tokenizer_model_name = tokenizer_model_name
        self.tokenizer_json_file = tokenizer_json_file
        self.special_tokens_map = special_tokens_map
        self.cache_dir = cache_dir
        self.deprecation_warnings = ({})
