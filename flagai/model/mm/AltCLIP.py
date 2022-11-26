from transformers.models.clip.modeling_clip import *
import torch.nn as nn
import torch
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers import CLIPProcessor
import os
from flagai.model.base_model import BaseModel

from .modeling_berts import BertSeriesConfig, RobertaSeriesConfig, BertSeriesModelWithTransformation, RobertaSeriesModelWithTransformation

STUDENT_CONFIG_DICT = {
    'hfl/chinese-roberta-wwm-ext': BertSeriesConfig,
    'hfl/chinese-roberta-wwm-ext-large': BertSeriesConfig,
    'xlm-roberta-large': RobertaSeriesConfig,
    'xlm-roberta-base': RobertaSeriesConfig,
    'bert-base-uncased': BertSeriesConfig,
    'bert': BertSeriesConfig,
    'xlm-roberta': RobertaSeriesConfig,
    'clip_text_model': BertSeriesConfig,
}

STUDENT_MODEL_DICT = {
    'hfl/chinese-roberta-wwm-ext': BertSeriesModelWithTransformation,
    'hfl/chinese-roberta-wwm-ext-large': BertSeriesModelWithTransformation,
    'xlm-roberta-large': RobertaSeriesModelWithTransformation,
    'xlm-roberta-base': RobertaSeriesModelWithTransformation,
    'bert': BertSeriesModelWithTransformation,
    'xlm-roberta': RobertaSeriesModelWithTransformation,
    'clip_text_model': BertSeriesModelWithTransformation,
}


@dataclass
class OursCLIPOutput(CLIPOutput):

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"
                                 ] else getattr(self, k).to_tuple()
            for k in self.keys())


class AltCLIPProcess(CLIPProcessor):
    tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")

    # tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # ## in some cases, we need to switch to different tokenizer
        # if tokenizer.vocab_size!=250002:
        #     print("tokenizer is not XLM-R, switched to BertTokenizer.")
        #     self.tokenizer = BertTokenizer.from_pretrained(tokenizer.name_or_path)


class AltCLIPProcessBert(CLIPProcessor):
    # tokenizer_class = ("XLMRobertaTokenizer","XLMRobertaTokenizerFast")
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # ## in some cases, we need to switch to different tokenizer
        # if tokenizer.vocab_size!=250002:
        #     print("tokenizer is not XLM-R, switched to BertTokenizer.")
        #     self.tokenizer = BertTokenizer.from_pretrained(tokenizer.name_or_path)


class AltCLIPConfig(CLIPConfig):

    def __init__(self,
                 text_model_name=None,
                 vision_model_name=None,
                 text_config_dict=None,
                 vision_config_dict=None,
                 projection_dim=512,
                 logit_scale_init_value=2.6592,
                 num_layers=3,
                 variant='invert',
                 **kwargs):
        super().__init__(text_config_dict, vision_config_dict, projection_dim,
                         logit_scale_init_value, **kwargs)
        if text_config_dict is None:
            text_config_dict = {}
        # when reload the config from local, we need name to select which class should be instanced.
        self.text_config = STUDENT_CONFIG_DICT[
            kwargs['text_config']['model_type']](**kwargs.pop('text_config'))
        self.num_layers = num_layers
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        self.variant = variant


class CLIPHF(CLIPPreTrainedModel):
    config_class = AltCLIPConfig

    def __init__(self, config: AltCLIPConfig, clip_model=None):
        super().__init__(config)

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}.")

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.project_dim
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = STUDENT_MODEL_DICT[text_config.model_type](
            text_config)

        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim,
                                           self.projection_dim,
                                           bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim,
                                         self.projection_dim,
                                         bias=False)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * self.config.logit_scale_init_value)

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = text_outputs['pooler_output']
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_text_features_diffusion(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_features = text_outputs['projection_state']
        # text_features = self.text_projection(pooled_output)

        return text_features

    def tokenize(self, texts, tokenizer, context_length: int = 77):

        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            all_tokens.append([tokenizer.vocab['<s>']] +
                              tokenizer.convert_tokens_to_ids(
                                  tokenizer.tokenize(text))[:context_length -
                                                            2] +
                              [tokenizer.vocab['</s>']])

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            assert len(tokens) <= context_length
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def encode(self,
               text,
               tokenizer,
               padding="max_length",
               truncation=True,
               max_length=77):
        device = next(self.text_model.parameters()).device
        text = tokenizer(text,
                         truncation=True,
                         max_length=77,
                         return_length=False,
                         return_overflowing_tokens=False,
                         padding="max_length",
                         return_tensors="pt")
        text["input_ids"] = torch.tensor(text["input_ids"]).to(device)
        text["attention_mask"] = torch.tensor(
            text['attention_mask']).to(device)
        # text = torch.tensor(text).to(device
        features = self.get_text_features_diffusion(**text)
        return features

    def get_text_encoder_out(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_out = text_outputs["sequence_out"]

        return sequence_out

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids=None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # text_embeds = self.text_model.get_text_embeds(text_outputs['pooler_output'],clip_outputs[1])
        text_embeds = text_outputs['pooler_output']
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(
            p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds,
                                       image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = clip_loss(logits_per_text)

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "text_embeds": text_embeds,
            "image_embeds": image_embeds,
            "text_model_output": text_outputs,
            "vision_model_output": vision_outputs,
            "logits": torch.cat([image_embeds, text_embeds], dim=-1)
        }


class AltCLIP(BaseModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    @classmethod
    def from_pretrain(cls,
                      download_path='./checkpoints/',
                      model_name='RoBERTa-base-ch',
                      only_download_config=False,
                      device="cpu",
                      **kwargs):
        super().download(download_path, model_name)
        pretrained_model_name_or_path = os.path.join(download_path, model_name)
        print(pretrained_model_name_or_path)
        return CLIPHF.from_pretrained(pretrained_model_name_or_path)
