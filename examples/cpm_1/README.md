CPM (Chinese Pre-trained Language Model) is a Transformer-based autoregressive language model, with 2.6 billion parameters and 100GB Chinese training data. The paper is available at https://arxiv.org/abs/2012.00413 


Pre-trained Language Models (PLMs) have
proven to be beneficial for various downstream
NLP tasks. Recently, GPT-3, with 175 billion parameters and 570GB training data, drew
a lot of attention due to the capacity of fewshot (even zero-shot) learning. However, applying GPT-3 to address Chinese NLP tasks
is still challenging, as the training corpus
of GPT-3 is primarily English, and the parameters are not publicly available. In this
technical report, we release the Chinese Pretrained Language Model (CPM) with generative pre-training on large-scale Chinese training data. To the best of our knowledge, CPM,
with 2.6 billion parameters and 100GB Chinese training data, is the largest Chinese pretrained language model, which could facilitate several downstream Chinese NLP tasks,
such as conversation, essay generation, cloze
test, and language understanding. Extensive
experiments demonstrate that CPM achieves
strong performance on many NLP tasks in
the settings of few-shot (even zero-shot) learning. The code and parameters are available at https://github.com/TsinghuaAI/CPMGenerate.

As example, CPM can complete the task of continuing ancient poems.

```python

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

if __name__ == '__main__':

    text = '''默写古诗:
    白日依山尽，黄河入海流。
    床前明月光，'''

    loader = AutoLoader(task_name="lm",
                        model_name="CPM-large-ch-generation",
                        model_dir="./state_dict/")

    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    predictor = Predictor(model=model,
                          tokenizer=tokenizer,
                          )

    out = predictor.predict_generate_randomsample(text, top_p=0.9, out_max_length=50)

    print(out)

```