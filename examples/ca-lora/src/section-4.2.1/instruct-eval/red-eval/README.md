# Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment

[**Paper**](https://arxiv.org/abs/2308.09662) | [**Github**](https://github.com/declare-lab/red-instruct) | [**Dataset**](https://huggingface.co/datasets/declare-lab/HarmfulQA) | [**Model**](https://huggingface.co/declare-lab/starling-7B)

**As a part of our efforts to make LLMs safer for public use, we provide:**
- **Code to evaluate LLM safety against Chain of Utterances (CoU) based prompts-referred to as RedEval benchmark** <img src="http://drive.google.com/uc?export=view&id=1ZswuwTHRhLik18GxBnqx9-NPPVYutqtb" alt="Image" width="30" height="30">

## Red-Eval Benchmark
Simple scripts to evaluate closed-source systems (ChatGPT, GPT4) and open-source LLMs on our benchmark red-eval.

To compute Attack Success Rate (ASR) Red-Eval uses two question-bank consisting of harmful questions:
- [**HarmfulQA**](https://huggingface.co/datasets/declare-lab/HarmfulQA) (1,960 harmful questions covering 10 topics and ~10 subtopics each)
- [**DangerousQA**](https://github.com/SALT-NLP/chain-of-thought-bias/blob/main/data/dangerous-q/toxic_outs.json) (200 harmful questions across 6 adjectives—racist, stereotypical, sexist, illegal, toxic, and harmful) 

### Installation
```
conda create --name redeval -c conda-forge python=3.11
conda activate redeval
pip install -r requirements.txt
```

### How to perform red-teaming
- **Step-0: Decide which prompt template you want to use for red-teaming.** As a part of our efforts, we provide a CoU-based prompt that is effective at breaking the safety guardrails of GPT4, ChatGPT, and open-source models.
  - [Chain of Utterances (CoU)](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/cou.txt)
  - [Chain of Thoughts (CoT)](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/cot.txt)
  - [Standard prompt](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/standard.txt)
  - [Suffix prompt](https://github.com/declare-lab/red-instruct/blob/main/red_prompts/suffix.txt)

    (_Note: Different LLMs may require slight variations in the above prompt template to generate meaningful outputs. To create a new template, you can refer to the above template files. Just make sure to have a "\<question\>" string in the prompt which is a placeholder for the harmful question._)
    
- **Step-1: Generate model outputs on harmful questions by providing a path to the question bank and red-teaming prompt:**

Closed-source models (GPT4 and ChatGPT):
```
  python generate_responses.py --model gpt4 --prompt red_prompts/cou.txt --dataset hamrful_questions/dangerousqa.json
  python generate_responses.py --model chatgpt --prompt red_prompts/cou.txt --dataset hamrful_questions/dangerousqa.json
```

  Open-source models:
  
```
  python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt red_prompts/cou.txt --dataset hamrful_questions/dangerousqa.json
```

  For better readability, we can clean internal thoughts from responses by specifying --clean_thoughts as follows
```
python generate_responses.py --model gpt4 --prompt red_prompts/cou.txt --dataset hamrful_questions/dangerousqa.json --clean_thoughts
python generate_responses.py --model chatgpt --prompt red_prompts/cou.txt --dataset hamrful_questions/dangerousqa.json --clean_thoughts
python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt red_prompts/cou.txt --dataset hamrful_questions/dangerousqa.json --clean_thoughts
```

To load models in 8-bit, we can specify --load_8bit as follows

```
  python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt red_prompts/cou.txt --dataset hamrful_questions/dangerousqa.json --load_8bit
```

- **Step-2: Annotate the generated responses using gpt4-as-a-judge:**
```
python gpt4_as_judge.py --response_file results/dangerousqa_gpt4_cou.json --save_path results
```

### Results
Attack Success Rate (ASR) of different red-teaming attempts.

|                | **(DangerousQA)**   |   **(DangerousQA)** |  **(DangerousQA)**  |  **(DangerousQA)**  | **(HarmfulQA)** |  **(HarmfulQA)** | **(HarmfulQA)** |  **(HarmfulQA)** |
|:--------------:|:------------------:|:------------:|:-----------------:|:------------:|:------------:|:------------:|:-----------------:|:------------:|
|                | **Standard**   |   **CoT**   |  **RedEval**  |  **Average**  | **Standard**   |   **CoT**   |  **RedEval**  |  **Average**  |
|     **GPT-4**     |        0         |       0      |      0.651      |     0.217     |       0        |      0.004     |      0.612      |     0.206     |
|    **ChatGPT**    |        0         |     0.005    |      0.728      |     0.244     |     0.018      |    0.027      |      0.728      |     0.257     |
|  **Vicuna-13B**   |     0.027      |     0.490    |      0.835      |     0.450     |       -        |      -        |       -        |       -       |
|  **Vicuna-7B** |     0.025      |     0.532    |      0.875      |     0.477     |       -        |      -        |       -        |       -       |
| **StableBeluga-13B** |     0.026      |     0.630    |      0.915      |     0.523     |       -        |      -        |       -        |       -       |
| **StableBeluga-7B** |     0.102      |     0.755    |      0.915      |     0.590     |       -        |      -        |       -        |       -       |
|**Vicuna-FT-7B**|     0.095      |     0.465    |      0.860      |     0.473     |       -        |      -        |       -        |       -       |
| **Llama2-FT-7B** |     0.722      |     0.860    |      0.896      |     0.826     |       -        |      -        |       -        |       -       |
|**Starling (Blue)** |     0.015      |     0.485    |      0.765      |     0.421     |       -        |      -        |       -        |       -       |
|**Starling (Blue-Red)** |     0.050      |     0.570    |      0.855      |     0.492     |       -        |      -        |       -        |       -       |
|     **Average**    |     0.116      |     0.479    |      0.830      |     0.471     |     0.010      |    0.016      |     0.67       |     0.232     |


## Citation

```bibtex
@misc{bhardwaj2023redteaming,
      title={Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment}, 
      author={Rishabh Bhardwaj and Soujanya Poria},
      year={2023},
      eprint={2308.09662},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
