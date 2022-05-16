## Introduction to Pattern-Exploiting Training (PET)

A pre-trained models somehow has the ability to 'understand' natural languages. However, primary pre-trained models can only perform the original
training task (like autoencoding and autoregressive task), which usually does not fit the various tasks we face in reality.
Therefore, we need to come up with a way to adapt the pre-trained models to downstream tasks.

Initially, an extra task-speficic architecture was added to the final layer of the pretrained-model, which is called finetuning. Nonetheless, the performance of
pretraining-fituning paradigm is not satisfatory especially when the pretraining task deviate from downstream tasks. As a result, [prompt-learning](https://arxiv.org/abs/2111.01998) is
proposed, where the core idea is to design a cloze-style task-specific prompt to combine pretrained-model and downstream tasks.

[Exploiting Cloze Questions for Few Shot Text Classification and Natural
Language Inference](https://arxiv.org/abs/2001.07676) proposed PET, which can reconstruct the input texts as cloze-style phrases, so that it is easier for pre-trained models to understand given tasks.

Due to different settings of input texts and labels among different tasks, we need to design appropriate cloze-style patterns. The paper introduced a sentiment classification example as shown below.

The given sentiment analysis task is: given an input context (eg. 'Best pizza ever!'), and limited categories (here there's 2 categories,
1 and -1 denotes positive and negative sentiment, respectively). Our model needs to automatically predict the probability of each
sentiment category given the context.

<div align=center><img src="img/pet_example.png"></div>

As shown in the picture, firstly the positive and negative labels are verbalized as 'great' and 'bad'. Then the pattern is designed as:

context + 'It was' + masked sentiment label + '.'

The probability for filling 'great' and 'bad' into the masked position will be returned by the finetuned model.

Here we can see that it requires users to design a pattern-verbalizer pair (PVP) for each given task.
In our project, this part is put in flagai/data/dataset/superglue/pvp.py. Under the class for a given task,
normally there exists two functions named 'verbalize' and 'get_parts', which represent the verbalizer and pattern design step.
An example is shown below. Note that for one task, there can be multiple patterns.

```
    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [[self.mask], ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [[self.mask], ' Question:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', [self.mask], ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', [self.mask], ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', [self.mask], ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [[self.mask], '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(
                self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YahooPVP.VERBALIZER[label]
```