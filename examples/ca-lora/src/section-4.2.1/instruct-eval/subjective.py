import numpy as np
import json
import random
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from modeling import select_model, SeqToSeqModel


class SubjectiveSample(BaseModel):
    Category: str
    Definition: str
    Prompt: str
    Answer: str = ""
    Review: str = ""
    Score: int = 0


class SubjectiveData(BaseModel):
    samples: List[SubjectiveSample]

    @classmethod
    def load(cls, path: str):
        samples = []

        try:
            with open(path) as f:
                for line in f:
                    samples.append(json.loads(line))

        except json.decoder.JSONDecodeError:
            data = pd.read_csv(path)
            data = data.fillna(value="")
            for raw in data.to_dict(orient="records"):
                samples.append(SubjectiveSample(**raw))

        print(dict(path=path, samples=len(samples)))
        return cls(samples=samples)

    @classmethod
    def load_from_huggingface(cls, path: str):
        samples = []
        for raw in load_dataset(path, split="train"):
            samples.append(SubjectiveSample(**raw))
        return cls(samples=samples)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.samples:
                print(s.json(), file=f)

    def analyze(self):
        random.seed(0)
        for s in random.sample(self.samples, k=8):
            print(s.json(indent=2))

        info = dict(
            samples=len(self.samples),
            category=Counter(s.Category for s in self.samples),
        )
        print(json.dumps(info, indent=2))


def test_data(path: str = "data/SubjectiveData.csv"):
    data = SubjectiveData.load(path)
    data.analyze()


def write_answers(
    folder: str, data_path: str = "declare-lab/InstructEvalImpact", **kwargs
):
    data = SubjectiveData.load_from_huggingface(data_path)
    model = select_model(max_input_length=512, max_output_length=1024, **kwargs)
    if isinstance(model, SeqToSeqModel):
        model.do_sample = True
    if model.model_path is None:
        model.load()

    path_out = Path(folder, Path(model.model_path).name, "samples.jsonl")
    print(dict(path_out=str(path_out)))

    sample: SubjectiveSample
    for sample in tqdm(data.samples, desc=data_path):
        sample.Answer = model.run(sample.Definition + " " + sample.Prompt)
        print(sample.json(indent=2))

    data.save(str(path_out))


def score_answers(mode: str, folder: str, **kwargs):
    model = select_model(max_input_length=1024, max_output_length=128, **kwargs)
    path_in = Path(folder, "samples.jsonl")
    data = SubjectiveData.load(str(path_in))
    path_out = Path(folder, mode).with_suffix(".jsonl")
    print(dict(path_out=path_out))

    if mode == "relevance":
        template = """Text: {text}
        
        Prompt: {prompt}
        
        How relevant is the text to the prompt? Select a suitable option number between 1 and 5 based on the options below.

        1. Inadequate: The text fails to provide any relevant information or insights related to the given prompt.
        2. Limited: The text may contain some relevant information, but significant gaps exist, and key aspects of the prompt are not adequately covered.
        3. Satisfactory: The text covers the main aspects of the prompt and provides relevant information, but it lacks depth and may not explore the topic in great detail.
        4. Proficient: The text provides a comprehensive response by addressing the key aspects of the prompt, offering relevant and well-supported information or arguments. 
        5. Excellent: The text thoroughly and thoughtfully addresses the prompt, demonstrating a comprehensive understanding of the topic. It offers insightful and original ideas, supported by relevant arguments and information.
        """
    elif mode == "coherence":
        # Prompt: {prompt}  # Don't include prompt as coherence is not prompt-specific
        template = """Text: {text}
        
        How coherent is the text? Select a suitable option number between 1 and 5 based on the options below.

        1. Inadequate: The text lacks logical organization, making it difficult to follow. Ideas are disjointed and phrased awkwardly, requiring significant effort to understand.
        2. Limited: The text demonstrates some attempt at organization, but there are significant gaps in coherence. Ideas may be loosely connected, and the arguments lack clarity.
        3. Satisfactory: The text generally follows a logical organization, but occasional disruptions or awkward phrasing may occur. There is an acceptable level of readability and understanding.
        4. Proficient: The text is clearly organized and easy to understand. Ideas and arguments flow smoothly, contributing to easy comprehension and a pleasant reading experience.
        5. Excellent: The text presents exceptionally coherent writing with a fluent and engaging flow of ideas, ensuring effortless comprehension and a delightful reading experience.
        """
    else:
        raise KeyError(mode)

    sample: SubjectiveSample
    for sample in tqdm(data.samples, desc=str(path_in)):
        text = template.format(text=sample.Answer, prompt=sample.Prompt)
        sample.Review = model.run(text)

        # Parse the first number in the output as the score
        for char in sample.Review:
            if char.isdigit():
                sample.Score = int(char)
        if not 1 <= sample.Score <= 5:
            sample.Score = 1

        print(sample.json(indent=2))

    scores = [sample.Score for sample in data.samples]
    print(dict(score=np.mean(scores), std=np.std(scores)))
    data.save(str(path_out))


def analyze_scores(pattern: str):
    for path in sorted(Path().glob(pattern)):
        data = SubjectiveData.load(str(path))
        df = pd.DataFrame([s.dict() for s in data.samples])
        print(path)
        print(df.groupby("Category")["Score"].mean())


"""
python subjective.py write_answers outputs/subjective --model_name openai --use_azure

python subjective.py write_answers outputs/subjective --model_name llama --model_path TheBloke/stable-vicuna-13B-HF --load_8bit

python subjective.py write_answers outputs/subjective --model_name seq_to_seq --model_path google/flan-t5-xxl --load_8bit

python subjective.py write_answers outputs/subjective --model_name seq_to_seq --model_path declare-lab/flan-alpaca-xxl --load_8bit

python subjective.py write_answers outputs/subjective --model_name causal --model_path databricks/dolly-v2-12b --load_8bit

python subjective.py write_answers outputs/subjective --model_name chatglm --model_path THUDM/chatglm-6b

python subjective.py write_answers outputs/subjective --model_name llama --model_path TheBloke/vicuna-13B-1.1-HF --load_8bit

################################################################################

python subjective.py score_answers relevance outputs/subjective/VisualQuestionAnswering --model_name openai --use_azure
{'score': 3.775, 'std': 0.8452070752188483}

python subjective.py score_answers relevance outputs/subjective/flan-t5-xxl --model_name openai --use_azure
{'score': 2.575, 'std': 1.0603655030224248}

python subjective.py score_answers relevance outputs/subjective/flan-alpaca-xxl --model_name openai --use_azure
{'score': 3.505, 'std': 0.8306473379238628}

python subjective.py score_answers relevance outputs/subjective/stable-vicuna-13B-HF --model_name openai --use_azure
{'score': 3.44, 'std': 0.8284926070883192}

python subjective.py score_answers relevance outputs/subjective/dolly-v2-12b --model_name openai --use_azure

python subjective.py score_answers relevance outputs/subjective/vicuna-13B-1.1-HF --model_name openai --model_path openai_info.json
{'score': 3.745, 'std': 0.7680983010005946}                                                                                    

p subjective.py analyze_scores "outputs/subjective/*/relevance.jsonl"
################################################################################

python subjective.py score_answers coherence outputs/subjective/VisualQuestionAnswering --model_name openai --use_azure
{'score': 3.925, 'std': 0.2817356917396161}

python subjective.py score_answers coherence outputs/subjective/flan-t5-xxl --model_name openai --use_azure
{'score': 3.145, 'std': 0.7169204976843666}

python subjective.py score_answers coherence outputs/subjective/flan-alpaca-xxl --model_name openai --use_azure
{'score': 3.46, 'std': 0.6069596362197407}

python subjective.py score_answers coherence outputs/subjective/stable-vicuna-13B-HF --model_name openai --use_azure
{'score': 3.205, 'std': 1.0968021699467958}

python subjective.py score_answers coherence outputs/subjective/dolly-v2-12b --model_name openai --use_azure

python subjective.py score_answers coherence outputs/subjective/vicuna-13B-1.1-HF --model_name openai --model_path openai_info.json
{'score': 3.815, 'std': 0.44807923406469086}                                                                                   

p subjective.py analyze_scores "outputs/subjective/*/coherence.jsonl"
"""


if __name__ == "__main__":
    Fire()
