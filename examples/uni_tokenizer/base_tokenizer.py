import logging
logger = logging.getLogger(__name__)
import os
from flagai.model.file_utils import _get_model_files
from flagai.data.tokenizer.glm_large_ch.glm_large_ch import get_encoder
from wp_tokenizer import WordpieceTokenizer

class BaseTokenizer(object):
    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        cache_dir=None, *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        vocab_file = 'vocab.txt'
        merges_file = 'merges.txt'
        sp_file = 'spm.model'
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'vocabs')

        tokenizer_class = ""
        # search the cache directory for certain files

        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            if "vocab.txt" in files:
                if "merges.txt" in files:
                    tokenizer_class = "bpe"
                else:
                    tokenizer_class = "wp"
            elif "spm.models" in files:
                tokenizer_class = "sp"
        if tokenizer_class == "":
            files = _get_model_files(pretrained_model_name_or_path)
            if "vocab.txt" in files:
                if "merges.txt" in files:
                    tokenizer_class = "bpe"
                else:
                    tokenizer_class = "wp"
            elif "spm.models" in files:
                tokenizer_class = "sp"
            else:
                raise FileNotFoundError("no tokenizer files")
        resolved_vocab_file = os.path.join(cache_dir, vocab_file)
        resolved_merges_file = os.path.join(cache_dir, merges_file)
        resolved_sp_file = os.path.join(cache_dir, sp_file)
        if tokenizer_class == "wp":
            return WordpieceTokenizer(vocab_file=resolved_vocab_file)
        elif tokenizer_class == "bpe":
            return cls(vocab_file=resolved_vocab_file, merges_file=resolved_merges_file, *inputs, **kwargs)
        elif tokenizer_class == "sp":
            return get_encoder(resolved_sp_file, "")

    def __init__(self, vocab_file=None, merges_file=None, sp_file=None,  *inputs, **kwargs):
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.sp_file = sp_file

if __name__ == '__main__':
    tokenizer = BaseTokenizer.from_pretrained('GLM-large-en')
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("fried chicken makes me happy")))