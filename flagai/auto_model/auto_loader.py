import importlib
from msilib.schema import Error
import os


class LazyImport(object):
    def __init__(self, name):
        self.cache = {}
        self.mod_name = name

    def __getattr__(self, name):
        mod = self.cache.get(self.mod_name)
        if not mod:
            mod = importlib.import_module(self.mod_name)
            self.cache[self.mod_name] = mod
        return getattr(mod, name)


ALL_TASK = {
    "bert_seq2seq": ["flagai.model.bert_model", "BertForSeq2seq"],
    "bert_masklm": ["flagai.model.bert_model", "BertForMaskLM"],
    "bert_sequence_labeling": ["flagai.model.bert_model", "BertForSequenceLabeling"],
    "bert_ner": ["flagai.model.bert_model", "BertForSequenceLabeling"],
    "bert_ner_crf": ["flagai.model.bert_model", "BertForSequenceLabelingCRF"],
    "bert_ner_gp": ["flagai.model.bert_model", "BertForSequenceLabelingGP"],
    "bert_embedding": ["flagai.model.bert_model", "BertForEmbedding"],
    "gpt2_seq2seq": ("flagai.model.gpt2_model", "GPT2Model"),
    "bert_classification": ["flagai.model.bert_model", "BertForClsClassifier"],
    "bert_semantic_matching": ["flagai.model.bert_model", "BertForClsClassifier"],
    "t5_seq2seq": ["flagai.model.t5_model", "T5Model"],
    "glm_seq2seq": ["flagai.model.glm_model", "GLMForSeq2Seq"],
    "glm_classification": ["flagai.model.glm_model", "GLMForSequenceClassification"]
}

MODEL_DICT = {
    "BERT-base-en": ["flagai.model.bert_model", "BertModel", "bert"],
    "RoBERTa-base-ch": ["flagai.model.bert_model", "BertModel", "bert"],
    "T5-base-en": ["flagai.model.t5_model", "T5Model", "t5"],
    "T5-base-ch": ["flagai.model.t5_model", "T5Model", "t5"],
    "GLM-large-ch": ["flagai.model.glm_model", "GLMModel", "glm"],
    "GLM-large-en": ["flagai.model.glm_model", "GLMModel", "glm"],
    "GPT2-base-ch": ["flagai.model.gpt2_model", "GPT2Model", "gpt2"],
}

TOKENIZER_DICT = {
    "BERT-base-en": ["flagai.data.tokenizer.bert.bert_tokenizer", "BertTokenizer"],
    "RoBERTa-base-ch": ["flagai.data.tokenizer.bert.bert_tokenizer", "BertTokenizer"],
    "T5-base-en": ["flagai.data.tokenizer.t5.t5_pegasus_tokenizer", "T5PegasusTokenizer"],
    "T5-base-ch": ["flagai.data.tokenizer.t5.t5_pegasus_tokenizer", "T5PegasusTokenizer"],
    "GLM-large-ch": [
        "flagai.data.tokenizer.glm_large_ch.glm_large_ch_tokenizer",
        "GLMLargeChTokenizer"
    ],
    "GLM-large-en": [
        "flagai.data.tokenizer.glm_large_en.glm_large_en_tokenizer",
        "GLMLargeEnTokenizer"
    ],
    "GPT2-base-ch": ["flagai.data.tokenizer.bert.bert_tokenizer", "BertTokenizer"],
}


class AutoLoader:
    def __init__(self,
                 task_name: str,
                 model_name: str,
                 model_dir: str = "./checkpoints/",
                 only_download_config: bool = False,
                 **kwargs):
        """
        Args:
            task_name: The task name, for example, "cls" for classification,
                      "sequence_labeling" for ner, part-of-speech tagging
                       and so on, "seq2seq" for sequence to sequence task.
            model_name: The model name, for example, "bert-base-chinese",
                        "RoBERTa-wwm-ext", "gpt2-chinese",
                        "t5-base-chinese" and so on.
            model_dir: The first level of the model download directory.
            load_pretrain_params: Whether to load the downloaded parameters.
            target_size: For the classification task, all labels size.
            inner_dim: For global pointer ner task, inner_dim is the
                       representation dim of start and end tokens.
        Examples::

            # load bert-base-chinese model and tokenizer to do the two
            # classification task of text.
            # Then the download path of config, model, vocab files is the
            # "./checkpoints/bert-base-chinese"
            >>> auto_loader = AutoLoader(task_name,
                                         model_name="bert-base-chinese",
                                         model_dir="checkpoints",
                                         load_pretrain_params=True,
                                         class_num=2)

        """
        if model_name not in MODEL_DICT:
            print(f"The model_name: {model_name} is not be supported")
            os._exit(0)
        brief_model_name = MODEL_DICT[model_name][2]
        # The dir to save config, vocab and model.
        download_path = os.path.join(model_dir, model_name)
        os.makedirs(download_path, exist_ok=True)

        vocab_file = os.path.join(model_dir, model_name, "vocab.txt")
        self.model = ALL_TASK.get(f"{brief_model_name}_{task_name}", None)
        if self.model is None:
            print(
                f"For the model_name: {model_name}, task_name: {task_name} \
                is not be supported."
            )
            os._exit(0)
        self.model = getattr(LazyImport(self.model[0]),
                                self.model[1]).from_pretrain(
                                    download_path=model_dir,
                                    model_name=f"{brief_model_name}_{task_name}",
                                    only_download_config=only_download_config,
                                    **kwargs)

        tokenizer_class = TOKENIZER_DICT[model_name]
        tokenizer_class = getattr(LazyImport(tokenizer_class[0]),
                                    tokenizer_class[1])
        self.tokenizer = tokenizer_class(vocab_file)




    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def load_pretrain_params(self, model_path):
        self.model.load_huggingface_weights(model_path)

        print(f"Loading done: {model_path}")
