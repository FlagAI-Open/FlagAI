# Seq2seq Method

Three ways are supported for sequence-to-sequence task generation, including:
1. Sequence-to-sequence generation tasks using only encoder models
2. Sequence-to-sequence generation tasks using only the decoder model
3. Sequence-to-sequence generation task with encoder+decoder model


## Encoder model
We provide the encoder model to perform the seq2seq task, for example, Bert, Roberta, GLM, and so on.

1. [Title Generation with RoBERTa model](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/roberta_title_generation)
2. [Title Generation with GLM model](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/glm_title_generation)

We add a special attention mask in the encoder model at training process. (https://github.com/microsoft/unilm)

![encoder_mask](./img/encoder_mask.png)

The inputs to this model are two sentences: [cls] sentence_1 [sep] sentence_2 [sep].

Where, sentence_1 does not use mask, and sentence_2 uses autoregressive mask.


## Decoder model

We also provide the decoder model for seq2seq task, such as gpt-2 models.

1. [Writing text with GPT-2 model](https://github.com/FlagAI-Open/FlagAI/blob/master/quickstart/writing_ch.py)

![decoder_mask](./img/decoder_mask.png)

Giving a start text, this model can be a good continuation of the text.

## Encoder-Decoder model
We also provide the encoder-decoder model for seq2seq task, such as T5 models.
1. [Title Generation with T5 model](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/t5_title_generation)


![encoder_decoder_mask](./img/encoder_decoder_mask.png)


Encoder only needs to encode once to get features, decoder continues to generate according to self and encoder features.