from argparse import Namespace

from fire import Fire
from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from modeling import select_model, EvalModel


def entry_point(
    problem_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )

    return results


def filter_code(completion: str, model: EvalModel) -> str:
    if "chatglm" in model.model_path:
        ## Remove boilerplate for the function
        return completion.split('"""\n')[-1].replace("`", "")
    else:
        ## The program tends to overwrite, we only take the first function
        completion = completion.lstrip("\n")
        return completion.split("\n\n")[0]


def gen_prompt(prompt: str, model: EvalModel) -> str:
    if "starcoder" in model.model_path:
        prompt = "<fim_prefix>" + prompt + "<fim_suffix><fim_middle>"
    else:
        prompt = (
            "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
            + prompt
        )
    if "starchat" in model.model_path:
        prompt = f"<|system|>\n<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>"
    return prompt


def count_indent(text: str) -> int:
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)
    return "\n".join(outputs)


def test_fix_indents():
    text = "   # TODO: Implement separate_paren_groups\nreturn []"
    print(fix_indents(text))


def evaluate(model: EvalModel, data_path: str, **kwargs) -> dict:
    dataset = read_problems(data_path)
    n_sample = kwargs.get("n_sample", 1)
    best_temperature = {1: 0.1, 10: 0.6, 100: 0.8}
    samples = []
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")
    for task_id in dataset:
        for i in range(n_sample):
            prompt = dataset[task_id]["prompt"]
            prompt = gen_prompt(prompt, model)
            temperature = best_temperature[n_sample]
            if temperature > 0:
                completion = model.run(prompt, temperature=temperature, do_sample=True)
            else:
                completion = model.run(prompt)

            completion = fix_indents(completion)
            sample = dict(task_id=task_id, completion=filter_code(completion, model))
            if i == 0:
                print("Prompt: ", "-" * 100)
                print(prompt)
                print("Completion: ", "-" * 100)
                print(filter_code(completion, model))
            samples.append(sample)
            progress_bar.update(1)
    progress_bar.close()

    model_name = model.model_path.replace("/", "_")
    pred_filename = f"humaneval_{model_name}_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(problem_file=data_path, sample_file=pred_filename)
    return result


def main(data_path: str = "human_eval/HumanEval.jsonl.gz", **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=1360, max_output_length=512, **kwargs)
    print(locals())

    result = evaluate(model, data_path, **kwargs)
    print(result)
    return result["pass@1"]


"""
p humaneval.py main  --model_name llama --model_path decapoda-research/llama-7b-hf --n_sample 1
{'pass@1': 0.105}

p humaneval.py main  --model_name llama --model_path chavinlo/alpaca-native --n_sample 1
{'pass@1': 0.105}

p humaneval.py main  --model_name llama --model_path eachadea/vicuna-13b --n_sample 1 --load_8bit
{'pass@1': 0.152}

python main.py humaneval --model_name llama --model_path decapoda-research/llama-7b-hf --n_sample 1 --load_8bit
{'pass@1': 0.10365853658536585}

python main.py humaneval --model_name llama --model_path decapoda-research/llama-13b-hf --n_sample 1 --load_8bit
{'pass@1': 0.12804878048780488}

python main.py humaneval --model_name llama --model_path huggyllama/llama-13b --n_sample 1 --load_8bit
{'pass@1': 0.12804878048780488}

python main.py humaneval --model_name causal --model_path Salesforce/codegen-6B-mono --n_sample 1
{'pass@1': 0.27439024390243905}

python main.py humaneval --model_name llama --model_path TheBloke/wizardLM-7B-HF --n_sample 1
{'pass@1': 0.1402439024390244}

python main.py humaneval --model_name seq_to_seq --model_path google/flan-t5-xl --n_sample 1
{'pass@1': 0.0}                                                        

python main.py humaneval --model_name causal --model_path stabilityai/stablelm-tuned-alpha-7b --n_sample 1
{'pass@1': 0.054878048780487805}

python main.py humaneval --model_name llama --model_path TheBloke/OpenAssistant-SFT-7-Llama-30B-HF --load_8bit
{'pass@1': 0.23170731707317074}

python main.py humaneval --model_name causal --model_path ../FlanPaca/export/flan-codegen-3b
{'pass@1': 0.15853658536585366}

python main.py humaneval --model_name llama --model_path huggyllama/llama-30b --load_8bit
{'pass@1': 0.1402439024390244}

python main.py humaneval --model_name causal --model_path facebook/opt-iml-30b --load_8bit --n_sample 1
{'pass@1': 0.09146341463414634}

"""


if __name__ == "__main__":
    Fire()
