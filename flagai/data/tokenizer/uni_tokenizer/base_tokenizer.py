import logging
logger = logging.getLogger(__name__)
import os
from flagai.model.file_utils import _get_model_files, _get_model_id, _get_vocab_path


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
        vocab_file = 'vocab.txt'
        merges_file = 'merges.txt'
        sp_model_file = 'spiece.model'
        if cache_dir is None:
            # cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs')
            cache_dir = "/root/.cache/FlagAI/"+tokenizer_model_name
        tokenizer_class = ""
        # search the cache directory for certain files

        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            if "spiece.model" in files:
                tokenizer_class = "sp"
            elif "vocab.txt" in files:
                if "merges.txt" in files:
                    tokenizer_class = "bpe"
                else:
                    tokenizer_class = "wp"

        if tokenizer_class == "":
            files = _get_model_files(tokenizer_model_name)
            model_id = _get_model_id(tokenizer_model_name)
            if "spiece.model" in files:
                tokenizer_class = "sp"
                _get_vocab_path(cache_dir + '/', sp_model_file, model_id, rank=0)
            elif "vocab.txt" in files:
                if "merges.txt" in files:
                    tokenizer_class = "bpe"
                    _get_vocab_path(cache_dir + '/', vocab_file, model_id, rank=0)
                    _get_vocab_path(cache_dir + '/', merges_file, model_id, rank=0)
                else:
                    tokenizer_class = "wp"
                    _get_vocab_path(cache_dir + '/', vocab_file, model_id, rank=0)
            else:
                raise FileNotFoundError("no tokenizer files")
        resolved_vocab_file = os.path.join(cache_dir, vocab_file)
        resolved_merges_file = os.path.join(cache_dir, merges_file)
        resolved_sp_file = os.path.join(cache_dir, sp_model_file)
        if tokenizer_class == "wp":
            return cls(vocab_file=resolved_vocab_file, tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name, cache_dir=cache_dir, *inputs, **kwargs)
        elif tokenizer_class == "bpe":
            return cls(vocab_file=resolved_vocab_file, merges_file=resolved_merges_file, tokenizer_class=tokenizer_class,
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
        # self.max_len = int(1e12)



if __name__ == '__main__':
    tokenizer = BaseTokenizer.from_pretrained('GLM-large-en')
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("fried chicken makes me happy")))