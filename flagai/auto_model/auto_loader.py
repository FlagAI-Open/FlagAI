# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import importlib
import os
import copy
from flagai.model.file_utils import _get_model_id


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


# 2 columns : 1-package name,  2-class name
ALL_TASK = {
    "bert_lm": ["flagai.model.bert_model", "BertModel"],
    "bert_seq2seq": ["flagai.model.bert_model", "BertForSeq2seq"],
    "bert_title-generation": ["flagai.model.bert_model", "BertForSeq2seq"],
    "bert_masklm": ["flagai.model.bert_model", "BertForMaskLM"],
    "bert_sequence-labeling":
        ["flagai.model.bert_model", "BertForSequenceLabeling"],
    "bert_sequence-labeling-crf":
        ["flagai.model.bert_model", "BertForSequenceLabeling"],
    "bert_sequence-labeling-gp":
        ["flagai.model.bert_model", "BertForSequenceLabeling"],
    "bert_ner": ["flagai.model.bert_model", "BertForSequenceLabeling"],
    "bert_ner-crf": ["flagai.model.bert_model", "BertForSequenceLabelingCRF"],
    "bert_ner-gp": ["flagai.model.bert_model", "BertForSequenceLabelingGP"],
    "bert_embedding": ["flagai.model.bert_model", "BertForEmbedding"],
    "bert_classification": ["flagai.model.bert_model", "BertForClsClassifier"],
    "bert_semantic-matching":
        ["flagai.model.bert_model", "BertForClsClassifier"],
    "gpt2_seq2seq": ("flagai.model.gpt2_model", "GPT2Model"),
    "gpt2_lm": ("flagai.model.gpt2_model", "GPT2Model"),
    "cpm_seq2seq": ("flagai.model.gpt2_model", "GPT2Model"),
    "cpm_lm": ("flagai.model.gpt2_model", "GPT2Model"),
    "t5_seq2seq": ["flagai.model.t5_model", "T5Model"],
    "t5_lm": ["flagai.model.t5_model", "T5Model"],
    "alm_lm": ["flagai.model.alm_model", "ALMModel"],
    "glm_lm": ["flagai.model.glm_model", "GLMModel"],
    "glm_seq2seq": ["flagai.model.glm_model", "GLMForSeq2Seq"],
    "glm_poetry": ["flagai.model.glm_model", "GLMForSeq2Seq"],
    "glm_classification":
        ["flagai.model.glm_model", "GLMForSequenceClassification"],
    "glm_title-generation": ["flagai.model.glm_model", "GLMForSeq2Seq"],
    "opt_seq2seq": ("flagai.model.opt_model", "OPTModel"),
    "opt_lm": ("flagai.model.opt_model", "OPTModel"),
    "vit_classification": ("flagai.model.vision.vit", "VisionTransformer"),
    "clip_txt_img_matching": ("flagai.model.mm.clip_model", "CLIP"),
    "swinv1_classification": ("flagai.model.vision.swinv1", "SwinTransformer"),
    "swinv2_classification": ("flagai.model.vision.swinv2",
                              "SwinTransformerV2"),
    "cpm3_lm": ("flagai.model.cpm3_model", "CPM3"),
    "cpm3_trian": ("flagai.model.cpm3_trian_model", "CPM3"),
    "diffusion_text2img": ("flagai.model.mm.AltDiffusion", "LatentDiffusion"),
    "altclip_txt_img_matching": ("flagai.model.mm.AltCLIP", "AltCLIP"),
    "evaclip_txt_img_matching": ("flagai.model.mm.eva_clip_model", "EVA_CLIP"),
}

# 4 columns : 1-package name,  2-class name, 3-model brief name, 4-model type
MODEL_DICT = {
    "bert-base-en": ["flagai.model.bert_model", "BertModel", "bert", "nlp"],
    "roberta-base-ch": ["flagai.model.bert_model", "BertModel", "bert", "nlp"],
    "t5-base-en": ["flagai.model.t5_model", "T5Model", "t5", "nlp"],
    "t5-base-ch": ["flagai.model.t5_model", "T5Model", "t5", "nlp"],
    "glm-large-ch": ["flagai.model.glm_model", "GLMModel", "glm", "nlp"],
    "alm-1.0": ["flagai.model.alm_model", "ALMModel", "alm", "nlp"],
    "glm-large-en": ["flagai.model.glm_model", "GLMModel", "glm", "nlp"],
    "gpt2-base-ch": ["flagai.model.gpt2_model", "GPT2Model", "gpt2", "nlp"],
    "cpm-large-ch": ["flagai.model.gpt2_model", "GPT2Model", "cpm", "nlp"],
    "opt-125m-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "opt-350m-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "opt-1.3b-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "opt-2.7b-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "opt-6.7b-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "opt-13b-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "opt-30b-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "opt-66b-en": ["flagai.model.opt_model", "OPTModel", "opt", "nlp"],
    "glm-10b-ch": ["flagai.model.glm_model", "GLMModel", "glm", "nlp"],
    "cpm3": ["flagai.model.cpm3_model", "CPM3", "cpm3", "nlp"],
    "cpm3-train": ["flagai.model.cpm3_train_model", "CPM3", "cpm3", "nlp"],
    "vit-base-p16-224":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "vit-base-p16-384":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "vit-base-p32-224":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "vit-base-p32-384":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "vit-large-p16-224":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "vit-large-p16-384":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "vit-large-p32-224":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "vit-large-p32-384":
        ["flagai.model.vision.vit", "VisionTransformer", "vit", "vision"],
    "clip-base-p32-224": ["flagai.model.mm.clip_model", "CLIP", "clip", "mm"],
    "clip-base-p16-224": ["flagai.model.mm.clip_model", "CLIP", "clip", "mm"],
    "clip-large-p14-224": ["flagai.model.mm.clip_model", "CLIP", "clip", "mm"],
    "clip-large-p14-336": ["flagai.model.mm.clip_model", "CLIP", "clip", "mm"],
    "clip-large-p14-336": ["flagai.model.mm.clip_model", "CLIP", "clip", "mm"],
    "altdiffusion":
    ["flagai.model.mm.diffusion", "LatentDiffusion", "diffusion", "mm","flagai.model.mm.AltCLIP", "AltCLIPProcess"],
    "altdiffusion-m9":
    ["flagai.model.mm.diffusion", "LatentDiffusion", "diffusion", "mm","flagai.model.mm.AltCLIP", "AltCLIPProcess"],
    "swinv1-base-patch4-window7-224":
        ["flagai.model.vision.swinv1", "SwinTransformer", "swinv1", "vision"],
    "swinv2-base-patch4-window8-256":
        ["flagai.model.vision.swinv2", "SwinTransformerV2", "swinv2", "vision"],
    "swinv2-base-patch4-window16-256":
        ["flagai.model.vision.swinv2", "SwinTransformerV2", "swinv2", "vision"],
    "swinv2-small-patch4-window16-256": [
        "flagai.model.vision.swinv2", "SwinTransformerV2", "swinv2", "vision"
    ],
    "altclip-xlmr-l": ["flagai.models.mm.AltCLIP", "AltCLIP", "altclip", "mm", "flagai.model.mm.AltCLIP",
                       "AltCLIPProcess"],
    "altclip-xlmr-l-m9": ["flagai.models.mm.AltCLIP", "AltCLIP", "altclip", "mm", "flagai.model.mm.AltCLIP",
                          "AltCLIPProcess"],
    "altclip-bert-b": ["flagai.models.mm.AltCLIP", "AltCLIP", "altclip", "mm", "flagai.model.mm.AltCLIP",
                       "AltCLIPProcessBert"],
    "eva-clip": ["flagai.model.mm.eva_clip_model", "EVA_CLIP", "evaclip", "mm"],
}


class AutoLoader:

    def __init__(self,
                 task_name: str = "lm",
                 model_name: str = "RoBERTa-base-ch",
                 model_dir: str = "./checkpoints/",
                 only_download_config: bool = False,
                 device="cpu",
                 **kwargs):
        """
        Args:
            task_name: The task name, for example, "cls" for classification,
                      "sequence_labeling" for ner, part-of-speech tagging
                       and so on, "seq2seq" for sequence to sequence task.
            model_name: The model name, for example, "BERT-base-ch",
                        "RoBERTa-base-ch", "GPT2-base-ch",
                        "T5-base-ch" and so on.
            model_dir: The first level of the model download directory.
            load_pretrain_params: Whether to load the downloaded parameters.
            target_size: For the classification task, all labels size.
            inner_dim: For global pointer ner task, inner_dim is the
                       representation dim of start and end tokens.
        Examples::

            # load BERT-base-ch model and tokenizer to do the two
            # classification task of text.
            # Then the download path of config, model, vocab files is the
            # "./checkpoints/BERT-base-ch"
            >>> auto_loader = AutoLoader(task_name,
                                         model_name="BERT-base-ch",
                                         model_dir="checkpoints",
                                         load_pretrain_params=True,
                                         class_num=2)

        """

        raw_model_name = copy.deepcopy(model_name)

        model_name = model_name.lower()

        if model_name not in MODEL_DICT:
            print(f"The model_name: {model_name} is not be supported")
            print(f"All supported models are {list(MODEL_DICT.keys())}")
            return

        brief_model_name = MODEL_DICT[model_name][2]
        model_type = MODEL_DICT[model_name][3]
        # The dir to save config, vocab and model.

        self.model_name = ALL_TASK.get(f"{brief_model_name}_{task_name}", None)
        if self.model_name is None:
            print(f"For the model_name: {model_name}, task_name: {task_name} \
                is not be supported.")
            tasks = self.get_task_name(brief_model_name)
            print(
                f"For the model_name: {model_name}, these tasks are be supported: {tasks}"
            )
            return

        download_path = os.path.join(model_dir, raw_model_name)
        print("*" * 20, task_name, model_name)

        model_name_ = self.is_exist_finetuned_model(raw_model_name, task_name)
        self.model = getattr(LazyImport(self.model_name[0]),
                             self.model_name[1]).from_pretrain(
            download_path=model_dir,
            model_name=model_name_,
            only_download_config=only_download_config,
            device=device,
            **kwargs)

        if model_type == "nlp":
            tokenizer_class = getattr(LazyImport("flagai.data.tokenizer"),
                                      "Tokenizer")
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, cache_dir=download_path)

        elif model_type == "mm":
            if model_name.startswith("altdiffusion"):
                self.process = getattr(LazyImport(MODEL_DICT[model_name][4]),
                                MODEL_DICT[model_name][5]).from_pretrained(os.path.join(model_dir, raw_model_name))
                self.tokenizer = self.process.tokenizer
                self.model.tokenizer = self.tokenizer
            elif "altclip" not in model_name:
                from flagai.data.tokenizer.clip.tokenizer import ClipTokenizer
                self.tokenizer = ClipTokenizer(bpe_path=os.path.join(download_path, 'bpe_simple_vocab_16e6.txt.gz'))
                self.transform = None
            else:
                
                self.process = getattr(LazyImport(MODEL_DICT[model_name][4]),
                                       MODEL_DICT[model_name][5]).from_pretrained(
                    os.path.join(model_dir, raw_model_name))
                self.transform = self.process.feature_extractor
                self.tokenizer = self.process.tokenizer

        else:
            self.tokenizer = None
            self.transform = None

    def is_exist_finetuned_model(self, raw_model_name, task_name):
        try:
            model_id = _get_model_id(f"{raw_model_name}-{task_name}")
            if model_id != 'null':
                model_name_ = f"{raw_model_name}-{task_name}"
                return model_name_
            else :
                return raw_model_name

        except:
            print("Model hub is not reachable.")
            return raw_model_name

    def get_task_name(self, brief_model_name):
        all_model_task = list(ALL_TASK.keys())
        model_tasks = [t for t in all_model_task if brief_model_name in t]
        tasks = [t.split("_")[-1] for t in model_tasks]
        tasks = list(set(tasks))
        return tasks

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def load_pretrain_params(self, model_path):
        self.model.load_huggingface_weights(model_path)
        print(f"Loading done: {model_path}")
