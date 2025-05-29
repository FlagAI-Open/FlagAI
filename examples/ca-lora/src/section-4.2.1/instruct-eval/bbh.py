from argparse import Namespace
from typing import List

from datasets import load_dataset, get_dataset_config_names
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from modeling import select_model, EvalModel


class BBHSample(BaseModel):
    input: str
    target: str

    def as_prompt(self, include_answer: bool = True):
        prompt = self.input
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(self.target)
        return prompt


class BBHData(BaseModel):
    samples: List[BBHSample]

    @classmethod
    def get_config_names(cls, path: str = "lukaemon/bbh") -> List[str]:
        return get_dataset_config_names(path)

    @classmethod
    def load_from_huggingface(
        cls, path: str = "lukaemon/bbh", config: str = "", split: str = "test"
    ):
        data = load_dataset(path, config, split=split)
        samples = [BBHSample(**raw) for raw in tqdm(data, desc=str((path, split)))]
        return cls(samples=samples)


def gen_prompt(data: BBHData, k=-1):
    prompt = ""
    if k == -1:
        k = len(data.samples)
    for i in range(k):
        prompt += data.samples[i].as_prompt()
    return prompt


def evaluate(model: EvalModel, data: BBHData, ntrain: int) -> dict:
    data_train = BBHData(samples=data.samples[:ntrain])
    data_test = BBHData(samples=data.samples[ntrain:])
    is_correct = []

    for i in range(len(data_test.samples)):
        # get prompt and make sure it fits
        k = int(ntrain)
        prompt_end = data_test.samples[i].as_prompt(include_answer=False)
        train_prompt = gen_prompt(data_train, k)
        prompt = train_prompt + prompt_end

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(data_train, k)
            prompt = train_prompt + prompt_end

        label = data_test.samples[i].target
        pred = model.run(prompt)
        is_correct.append(pred.strip().startswith(label))
        if i == 0:
            print(dict(prompt=prompt, label=label, pred=pred))

    return dict(score=sum(is_correct) / len(is_correct))


def main(data_dir: str = "lukaemon/bbh", ntrain: int = 3, **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=2048, max_output_length=32, **kwargs)
    print(locals())

    all_results = []
    for name in tqdm(BBHData.get_config_names()):
        data = BBHData.load_from_huggingface(config=name)
        result = evaluate(model, data, ntrain=ntrain)
        all_results.append(result)
        print(dict(name=name, **result))

    score = sum(res["score"] for res in all_results) / len(all_results)
    print(dict(average=score))
    return score


"""
p bbh.py main "lukaemon/bbh" --model_name seq_to_seq --model_path google/flan-t5-xl 
{'average': 0.40261571422898645}

p bbh.py main "lukaemon/bbh" --model_name llama --model_path decapoda-research/llama-7b-hf
{'average': 0.30963361708212966}

p bbh.py main "lukaemon/bbh" --model_name llama --model_path chavinlo/alpaca-native
{'average': 0.3335667396422546}

p bbh.py main "lukaemon/bbh" --model_name chatglm --model_path THUDM/chatglm-6b
{'average': 0.31384628677534854}

python main.py bbh --model_name llama --model_path chavinlo/alpaca-13b --load_8bit
{'average': 0.33351335206026284}

python main.py bbh --model_name causal --model_path togethercomputer/Pythia-Chat-Base-7B
{'average': 0.29975163365323554}

python main.py bbh --model_name llama --model_path decapoda-research/llama-13b-hf --load_8bit
{'average': 0.3719930899679183}

python main.py bbh --model_name llama --model_path TheBloke/koala-7B-HF --load_8bit
{'average': 0.3118093830908477}

python main.py bbh --model_name llama --model_path TheBloke/koala-13B-HF --load_8bit
{'average': 0.3468942926723247}

python main.py bbh --model_name llama --model_path eachadea/vicuna-13b --load_8bit
{'average': 0.3717117791946168}

python main.py bbh --model_name causal --model_path togethercomputer/GPT-NeoXT-Chat-Base-20B --load_8bit
{'average': 0.30625775783670517}

python main.py bbh --model_name seq_to_seq --model_path google/flan-t5-xxl --load_8bit
{'average': 0.4391247239073324}

python main.py bbh --model_name seq_to_seq --model_path declare-lab/flan-alpaca-xl
{'average': 0.27024358682253424}

python main.py bbh --model_name causal --model_path databricks/dolly-v2-12b --load_8bit
{'average': 0.3003781793255478}

python main.py bbh --model_name llama --model_path wombat-7b-gpt4
{'average': 0.32478557123866053}

python main.py bbh --model_name causal --model_path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --load_8bit
{'average': 0.3008946837550956}

python main.py bbh --model_name seq_to_seq --model_path declare-lab/flan-alpaca-gpt4-xl
{'average': 0.3481774107746648}

python main.py bbh --model_name seq_to_seq --model_path google/flan-t5-xl --lora_path declare-lab/flan-alpaca-xl-lora
{'average': 0.280621115472374}

python main.py bbh --model_name causal --model_path stabilityai/stablelm-base-alpha-7b
{'average': 0.27506399778710994}

python main.py bbh --model_name llama --model_path huggyllama/llama-30b --load_8bit
{'average': 0.39346261713538594}

python main.py bbh --model_name llama --model_path huggyllama/llama-13b --load_8bit
{'average': 0.3719930899679183}

python main.py bbh --model_name causal --model_path Salesforce/codegen-6B-mono
{'average': 0.29238637284403873}

python main.py bbh --model_name llama --model_path TheBloke/wizardLM-7B-HF --load_8bit
{'average': 0.32918965812558487}

python main.py bbh --model_name causal --model_path ../FlanPaca/export/flan-opt-3b
{'average': 0.2885727015589716}

python main.py bbh --model_name causal --model_path ../FlanPaca/export/alpaca-opt-3b
{'average': 0.29839096448936264}

python main.py bbh --model_name causal --model_path facebook/opt-2.7b
{'average': 0.2883221490772978}

python main.py bbh --model_name seq_to_seq --model_path bigscience/T0pp --load_8bit
{'average': 0.10846421143903981}

python main.py bbh --model_name openai --model_path VisualQuestionAnswering --use_azure
{'average': 0.49579194980796804}

python main.py bbh --model_name seq_to_seq --model_path bigscience/T0pp --load_8bit
{'average': 0.10846421143903981}

python main.py bbh --model_name llama --model_path TheBloke/OpenAssistant-SFT-7-Llama-30B-HF --load_8bit
{'average': 0.3928688114157221}

python main.py bbh --model_name causal --model_path stabilityai/stablelm-tuned-alpha-7b
{'average': 0.2892898981686167}

python main.py bbh --model_name causal --model_path bigscience/bloomz-7b1
{'average': 0.2527831178060011}

python main.py bbh --model_name seq_to_seq --model_path google/flan-ul2 --load_8bit
{'average': 0.4479251941380086}

python main.py bbh --model_name causal --model_path facebook/opt-iml-30b --load_8bit
{'average': 0.31348283464988275}

python main.py bbh --model_name seq_to_seq --model_path declare-lab/flan-alpaca-xxl --load_8bit
{'average': 0.23395300775163477}

"""


if __name__ == "__main__":
    Fire()
