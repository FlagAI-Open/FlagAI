from transformers import PreTrainedTokenizerFast


from ..tokenizer import CommandToken, Tokenizer

class GalacticaTokenizer(Tokenizer):
    def __init__(self, download_dir) -> None:
        pass
        self.text_tokenizer = PreTrainedTokenizerFast.from_pretrained(download_dir)
         # parse tokens and vocabs from tokenizer
        self._tokens = list(self.text_tokenizer.get_vocab().keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.get_vocab().items()}
        self.num_tokens = len(self._tokens)

        self._command_tokens = [
            CommandToken('pad', '[PAD]', self.get_specialid_from_text_tokenizer('pad')),
            CommandToken('cls', '[CLS]', self.get_specialid_from_text_tokenizer('cls')),
            CommandToken('MASK', '[MASK]',
                         self.get_specialid_from_text_tokenizer('mask')),
            CommandToken('unk', '[UNK]', self.get_specialid_from_text_tokenizer('unk')),
            CommandToken('sep', '[SEP]', self.get_specialid_from_text_tokenizer('sep')),
            CommandToken('eos', '[PAD]', self.get_specialid_from_text_tokenizer('pad')),
        ]
        
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

    def get_specialid_from_text_tokenizer(self, token):
        if token in ["eos", "sep"]:
            return self._vocab.get('</s>')
        elif token == "cls":
            return self._vocab.get('<s>')
        elif token == "unk":
            return self._vocab.get('<unk>')
        elif token == "pad":
            return self._vocab.get('<pad>')
        elif token == "mask":
            return self._vocab.get('<mask>')
        else:
            raise NameError("token not exists")

    def encode_plus(self, text, max_length=512):
        return self.text_tokenizer.encode_plus(text, truncation=True, max_length=max_length)

    def decode(self, ids):
        return self.text_tokenizer.decode(ids)

    def get_vocab(self):
        return self.text_tokenizer.get_vocab()
    
    def get_command_id(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name].Id

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def encode_plus(self,
                    text,
                    second_text=None,
                    truncation=False,
                    max_length=None,):
        
        return self.text_tokenizer.encode_plus(text, 
                                          text_pair=second_text, 
                                          truncation=truncation, 
                                          max_length=max_length,
                                          add_special_tokens=True)

    def tokenize(self, **kwargs):
        return self.text_tokenizer.tokenize(**kwargs)
    
    def __len__(self):
        return len(self.text_tokenizer)

if __name__ == "__main__":
    pass 
        

