import random
from collections import Counter
from pathlib import Path
from typing import List

from fire import Fire
from pydantic import BaseModel
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from modeling import select_model, EvalModel


class CrassSample(BaseModel):
    premise: str
    question: str
    options: List[str]
    answer: str

    def as_prompt(self, include_answer=True):
        prompt = self.premise.strip() + " " + self.question.strip()
        labels = list("ABCD")
        for i, o in enumerate(self.options):
            prompt += f"\n{labels[i]}. {o}"
        prompt += "\nAnswer:"

        if include_answer:
            prompt = f"{prompt} {self.get_answer_label()}\n\n"

        return prompt

    def get_answer_label(self) -> str:
        labels = list("ABCD")
        return labels[self.options.index(self.answer)]


class CrassData(BaseModel):
    samples: List[CrassSample]

    @classmethod
    def load_train_set(cls):
        # From few-shot samples in paper: https://aclanthology.org/2022.lrec-1.229/
        samples = [
            CrassSample(
                premise="A feather falls from a skyscraper.",
                question="What would have happened if a computer had fallen from the skyscraper?",
                options=[
                    "The computer would have remained intact.",
                    "That is not possible.",
                    "The computer would have been crushed.",
                ],
                answer="The computer would have been crushed.",
            ),
            CrassSample(
                premise="A lightning hits a tree.",
                question="What would have happened if a marble would have hit the tree?",
                options=[
                    "It would have burned down.",
                    "Nothing special would have happened.",
                    "The tree would have kissed the lightning.",
                ],
                answer="Nothing special would have happened.",
            ),
            CrassSample(
                premise="A man drinks a beer.",
                question="What would have happened if the man had drunk a rainbow?",
                options=[
                    "It would have been tasty.",
                    "It would have been awful.",
                    "That is not possible.",
                ],
                answer="That is not possible.",
            ),
        ]

        return cls(samples=samples)

    @classmethod
    def load_test_set(
        cls,
        path: str = "https://raw.githubusercontent.com/apergo-ai/CRASS-data-set/main/CRASS_FTM_main_data_set.csv",
        seed: int = 0,
    ):
        if not Path(Path(path).name).exists():
            download_url(path, root=".")

        samples = []
        random.seed(seed)
        with open(Path(path).name) as f:
            f.readline()
            for line in f:
                _, _, premise, question, *options = line.strip().split(";")
                options = [o.strip() for o in options[:4] if o.strip()]
                answer = options[0]
                random.shuffle(options)
                samples.append(
                    CrassSample(
                        premise=premise,
                        question=question,
                        options=options,
                        answer=answer,
                    )
                )

        return cls(samples=samples)

    def analyze(self):
        random.seed(0)
        for sample in random.sample(self.samples, k=3):
            print(sample.json(indent=2))
        for sample in self.samples:
            assert sample.answer in sample.options

        info = dict(
            samples=len(self.samples),
            num_options=Counter(len(s.options) for s in self.samples),
            labels=Counter(s.get_answer_label() for s in self.samples),
        )
        print(info)


def test_data():
    data = CrassData.load_train_set()
    data.analyze()
    data = CrassData.load_test_set()
    data.analyze()


def gen_prompt(data: CrassData, k=-1):
    prompt = ""
    if k == -1:
        k = len(data.samples)
    for sample in data.samples[:k]:
        prompt += sample.as_prompt()
    return prompt


def evaluate(model: EvalModel, data_train: CrassData, data_test: CrassData) -> dict:
    is_correct = []
    score = 0

    progress = tqdm(data_test.samples)
    sample: CrassSample
    for sample in progress:
        # get prompt and make sure it fits
        k = int(len(data_train.samples))
        prompt_end = sample.as_prompt(include_answer=False)
        train_prompt = gen_prompt(data_train, k)
        prompt = train_prompt + prompt_end

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(data_train, k)
            prompt = train_prompt + prompt_end

        label = sample.get_answer_label()
        pred = model.run(prompt).strip()
        is_correct.append(pred.startswith(label))
        score = sum(is_correct) / len(is_correct)
        progress.set_postfix(score=score)
        print(dict(prompt=prompt, label=label, pred=pred))

    return dict(score=score)


def main(ntrain: int = 3, **kwargs):
    model = select_model(max_input_length=2048, max_output_length=8, **kwargs)
    print(locals())

    all_results = []
    data_train = CrassData.load_train_set()
    data_train.samples = data_train.samples[:ntrain]
    data_test = CrassData.load_test_set()
    data_test.analyze()
    result = evaluate(model, data_train, data_test)
    print(result)
    return result["score"]


"""
python main.py crass --model_name seq_to_seq --model_path bigscience/T0pp --load_8bit
{'crass': 58.03}

python main.py crass --model_name seq_to_seq --model_path google/flan-t5-xl
{'crass': 91.24}

python main.py crass --model_name seq_to_seq --model_path declare-lab/flan-alpaca-xxl --load_8bit
{'crass': 90.15}

python main.py crass --model_name llama --model_path TheBloke/stable-vicuna-13B-HF --load_8bit
{'crass': 67.52}

python main.py crass --model_name causal --model_path mosaicml/mpt-7b
{'crass': 39.42}

python main.py crass --model_name causal --model_path mosaicml/mpt-7b-instruct
{'crass': 38.32}

python main.py crass --model_name causal --model_path mosaicml/mpt-7b-chat
{'crass': 47.45}

python main.py crass --model_name llama --model_path huggyllama/llama-30b --load_8bit
{'crass': 68.61}

python main.py crass --model_name llama --model_path chavinlo/alpaca-native
{'crass': 50.73}

python main.py crass --model_name rwkv --model_path https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-7B-v11-Eng99%25-Other1%25-20230427-ctx8192.pth
{'crass': 28.47}

python main.py crass --model_name rwkv --model_path https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-14B-v11x-Eng99%25-Other1%25-20230501-ctx8192.pth
{'crass': 31.75}

python main.py crass --model_name openai --model_path VisualQuestionAnswering --use_azure
{'crass': 90.51}

"""


if __name__ == "__main__":
    Fire()
