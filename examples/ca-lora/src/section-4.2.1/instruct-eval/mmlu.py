"""
Adapted from https://github.com/hendrycks/test/blob/master/evaluate_flan.py
"""

import os
from argparse import Namespace

import numpy as np
import pandas as pd
from fire import Fire
from tqdm import tqdm

from modeling import select_model, EvalModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(get_choices()[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def evaluate(args, subject, model: EvalModel, dev_df, test_df):
    cors = []
    all_probs = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        pred = model.run(prompt)
        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(data_dir: str = "data/mmlu", ntrain: int = 5, **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=2048, max_output_length=2, **kwargs)
    print(locals())

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in get_subcategories().values()
        for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = evaluate(args, subject, model, dev_df, test_df)
        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories().keys():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    return weighted_acc


"""
p mmlu.py main data/mmlu --model_name seq_to_seq --model_path declare-lab/flan-alpaca-xl
0.46560319042871384

p mmlu.py main data/mmlu --model_name seq_to_seq --model_path ../FlanPaca/flan-alpaca-xl-epoch-1
0.45292693348525853

p mmlu.py main data/mmlu --model_name seq_to_seq --model_path google/flan-t5-base 
0.3404785643070788

p mmlu.py main data/mmlu --model_name seq_to_seq --model_path google/flan-t5-xl 
0.49252243270189433

p mmlu.py main data/mmlu --model_name causal --model_path facebook/opt-iml-max-1.3b
0.2756017661301809

p mmlu.py main data/mmlu --model_name causal --model_path EleutherAI/gpt-j-6B
0.2714713003845606

p mmlu.py main data/mmlu --model_name llama --model_path decapoda-research/llama-7b-hf
0.35215781227745335

p mmlu.py main data/mmlu --model_name llama --model_path chavinlo/alpaca-native
0.4163936761145136

p mmlu.py main data/mmlu --model_name chatglm --model_path THUDM/chatglm-6b
0.36155818259507194

python main.py mmlu --model_name llama --model_path chavinlo/alpaca-13b --load_8bit
Average accuracy: 0.425

python main.py mmlu --model_name seq_to_seq --model_path google/flan-t5-xxl --load_8bit
Average accuracy: 0.545

python main.py mmlu --model_name causal --model_path togethercomputer/Pythia-Chat-Base-7B
Average accuracy: 0.268

python main.py mmlu --model_name llama --model_path decapoda-research/llama-13b-hf --load_8bit
Average accuracy: 0.462

python main.py mmlu --model_name llama --model_path TheBloke/koala-7B-HF --load_8bit
Average accuracy: 0.250

python main.py mmlu --model_name llama --model_path TheBloke/koala-13B-HF --load_8bit
Average accuracy: 0.446

python main.py mmlu --model_name llama --model_path eachadea/vicuna-13b --load_8bit
Average accuracy: 0.497

python main.py mmlu --model_name causal --model_path databricks/dolly-v2-12b --load_8bit
Average accuracy: 0.257

python main.py mmlu --model_name llama --model_path wombat-7b-gpt4
Average accuracy: 0.330

python main.py mmlu --model_name seq_to_seq --model_path declare-lab/flan-alpaca-gpt4-xl
Average accuracy: 0.456

python main.py mmlu --model_name llama --model_path huggyllama/llama-7b --lora_path tloen/alpaca-lora-7b
Average accuracy: 0.359

python main.py mmlu --model_name llama --model_path huggyllama/llama-7b --lora_path tloen/alpaca-lora-7b --load_8bit
Average accuracy: 0.355

python main.py mmlu --model_name llama --model_path huggyllama/llama-7b --lora_path chansung/gpt4-alpaca-lora-7b
Average accuracy: 0.356

python main.py mmlu --model_name llama --model_path huggyllama/llama-13b --lora_path chansung/gpt4-alpaca-lora-13b --load_8bit
Average accuracy: 0.464

python main.py mmlu --model_name seq_to_seq --model_path google/flan-t5-xl --lora_path declare-lab/flan-alpaca-xl-lora
Average accuracy: 0.493

python main.py mmlu --model_name seq_to_seq --model_path bigscience/mt0-xl
Average accuracy: 0.304

python main.py mmlu --model_name causal --model_path OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --load_8bit
Average accuracy: 0.270

python main.py mmlu --model_name causal --model_path stabilityai/stablelm-base-alpha-7b
Average accuracy: 0.262

python main.py mmlu --model_name llama --model_path huggyllama/llama-30b --load_8bit
Average accuracy: 0.578                                                                                                                                                                        

python main.py mmlu --model_name llama --model_path huggyllama/llama-13b --load_8bit
Average accuracy: 0.462

python main.py mmlu --model_name causal --model_path Salesforce/codegen-6B-mono
Average accuracy: 0.261

python main.py mmlu --model_name llama --model_path TheBloke/wizardLM-7B-HF --load_8bit
Average accuracy: 0.364

python main.py mmlu --model_name causal --model_path facebook/opt-2.7b
Average accuracy: 0.257

python main.py mmlu --model_name seq_to_seq --model_path declare-lab/flan-sharegpt-xl
Average accuracy: 0.446

python main.py mmlu --model_name causal --model_path ../FlanPaca/export/flan-opt-3b
Average accuracy: 0.288

python main.py mmlu --model_name causal --model_path ../FlanPaca/export/alpaca-opt-3b
Average accuracy: 0.263

python main.py mmlu --model_name seq_to_seq --model_path bigscience/T0pp --load_8bit
Average accuracy: 0.368

python main.py mmlu --model_name seq_to_seq --model_path google/t5-xl-lm-adapt
Average accuracy: 0.233

python main.py mmlu --model_name llama --model_path TheBloke/OpenAssistant-SFT-7-Llama-30B-HF --load_8bit
Average accuracy: 0.569

python main.py mmlu --model_name causal --model_path stabilityai/stablelm-tuned-alpha-7b
Average accuracy: 0.244

python main.py mmlu --model_name causal --model_path bigscience/bloomz-7b1
Average accuracy: 0.372

python main.py mmlu --model_name seq_to_seq --model_path google/flan-ul2 --load_8bit
Average accuracy: 0.550

python main.py mmlu --model_name causal --model_path ../FlanPaca/export/flan-codegen-3b
Average accuracy: 0.294

python main.py mmlu --model_name llama --model_path TheBloke/stable-vicuna-13B-HF --load_8bit
Average accuracy: 0.492

"""


if __name__ == "__main__":
    Fire()
