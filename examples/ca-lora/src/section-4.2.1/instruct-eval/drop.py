import json
import random
from pathlib import Path
from typing import List

from datasets import load_dataset
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from modeling import select_model, EvalModel


class DropAnswer(BaseModel):
    spans: List[str]
    types: List[str]


class DropSample(BaseModel):
    section_id: str
    query_id: str
    passage: str
    question: str
    answers_spans: DropAnswer

    def get_answers(self) -> List[str]:
        return [text.strip() for text in self.answers_spans.spans]

    def as_prompt(self, include_answer=True):
        prompt = self.passage.strip() + " " + self.question.strip()
        prompt += "\nAnswer:"
        if include_answer:
            answer = self.get_answers()[0]
            prompt = f"{prompt} {answer}\n\n"
        return prompt


class DropData(BaseModel):
    samples: List[DropSample]

    @classmethod
    def load_from_huggingface(cls, path: str = "drop", split: str = "validation"):
        data = load_dataset(path, split=split)
        samples = [DropSample(**raw) for raw in tqdm(data, desc=str((path, split)))]
        return cls(samples=samples)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            samples = [DropSample(**json.loads(line)) for line in f]
        return cls(samples=samples)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.samples:
                print(s.json(), file=f)

    def analyze(self):
        info = dict(samples=len(self.samples))
        print(json.dumps(info, indent=2))

    def train_test_split(self, num_train: int, seed: int = 0):
        random.seed(seed)
        samples_train = random.sample(self.samples, k=num_train)
        texts_train = set(s.passage for s in samples_train)
        samples_test = [s for s in self.samples if s.passage not in texts_train]
        print(dict(samples_train=len(samples_train), samples_test=len(samples_test)))
        return DropData(samples=samples_train), DropData(samples=samples_test)


def test_data(path_out: str = "data/drop.json"):
    data = DropData.load_from_huggingface()
    data.analyze()
    data.save(path_out)


def gen_prompt(data: DropData, k=-1):
    prompt = ""
    if k == -1:
        k = len(data.samples)
    for sample in data.samples[:k]:
        prompt += sample.as_prompt()
    return prompt


def filter_dataset(data: DropData) -> DropData:
    samples = []
    for s in data.samples:
        spans = s.get_answers()
        if len(set(spans)) == 1:
            samples.append(s)

    print(dict(orig=len(data.samples), after_filter=len(samples)))
    return DropData(samples=samples)


def evaluate(model: EvalModel, data: DropData, ntrain: int) -> dict:
    data = filter_dataset(data)
    data_train, data_test = data.train_test_split(num_train=ntrain)
    is_correct = []
    score = 0

    progress = tqdm(data_test.samples)
    sample: DropSample
    for sample in progress:
        # get prompt and make sure it fits
        k = int(ntrain)
        prompt_end = sample.as_prompt(include_answer=False)
        train_prompt = gen_prompt(data_train, k)
        prompt = train_prompt + prompt_end

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(data_train, k)
            prompt = train_prompt + prompt_end

        label = sample.get_answers()[0]
        pred = model.run(prompt).strip()
        is_correct.append(pred.startswith(label))
        score = sum(is_correct) / len(is_correct)
        progress.set_postfix(score=score)
        print(dict(prompt=prompt, label=label, pred=pred))

    return dict(score=score)


def main(data_dir: str = "drop", ntrain: int = 3, **kwargs):
    model = select_model(max_input_length=2048, max_output_length=8, **kwargs)
    print(locals())

    all_results = []
    data = DropData.load_from_huggingface()
    result = evaluate(model, data, ntrain=ntrain)
    print(result)
    return result["score"]


"""
python main.py drop --model_name seq_to_seq --model_path google/flan-t5-xl
{'score': 0.5632458233890215}

python main.py drop --model_name llama --model_path eachadea/vicuna-13b --load_8bit
{'score': 0.3291964996022275}

python main.py drop --model_name llama --model_path chavinlo/alpaca-native
{'score': 0.2638027048528242}

python main.py drop --model_name llama --model_path decapoda-research/llama-13b-hf --load_8bit
{'score': 0.35338106603023073}

python main.py drop --model_name llama --model_path TheBloke/koala-13B-HF --load_8bit
{'score': 0.28353221957040575}

python main.py drop --model_name llama --model_path decapoda-research/llama-7b-hf
{'score': 0.27653142402545744}

python main.py drop --model_name chatglm --model_path THUDM/chatglm-6b
{'score': 0.44200477326968973}

python main.py drop --model_name seq_to_seq --model_path declare-lab/flan-alpaca-gpt4-xl
{'score': 0.28496420047732696}

python main.py drop --model_name seq_to_seq --model_path google/flan-t5-xl --lora_path declare-lab/flan-alpaca-xl-lora
{'score': 0.36945107398568017}

python main.py drop --model_name llama --model_path TheBloke/wizardLM-7B-HF
{'score': 0.2587112171837709}

python main.py drop --model_name seq_to_seq --model_path google/flan-t5-xxl --load_8bit
{'score': 0.6728719172633254}

python main.py drop --model_name causal --model_path Salesforce/codegen-6B-mono
{'score': 0.1280827366746221}

python main.py drop --model_name seq_to_seq --model_path bigscience/T0pp --load_8bit
{'score': 0.016070007955449484}

python main.py drop --model_name llama --model_path TheBloke/OpenAssistant-SFT-7-Llama-30B-HF --load_8bit
{'score': 0.46046141607000796}

python main.py drop --model_name causal --model_path stabilityai/stablelm-tuned-alpha-7b
{'score': 0.1271280827366746}

python main.py drop --model_name causal --model_path bigscience/bloomz-7b1
{'score': 0.23182179793158314}

python main.py drop --model_name llama --model_path huggyllama/llama-30b --load_8bit
{'score': 0.45425616547334924}

python main.py drop --model_name causal --model_path facebook/opt-iml-30b --load_8bit
{'score': 0.4758949880668258}

python main.py drop --model_name seq_to_seq --model_path declare-lab/flan-alpaca-xxl --load_8bit
{'score': 0.6233890214797136}

"""


if __name__ == "__main__":
    Fire()
