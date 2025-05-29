import os
import time
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', help='prompt template to be used for red-teaming', type=str, required=True)
parser.add_argument('--clean_thoughts', help='remove internal thoughts from the output', action='store_true', required=False)
parser.add_argument('--model', help='model under evaluation: gpt4, chatgpt, huggingface_model_path', type=str, required=True)
parser.add_argument('--save_path', help='path where the model results to be saved', type=str, required=False, default='results')
parser.add_argument('--num_samples', help='number of first num_samples to test from the dataset', type=int, required=False, default=-1)
parser.add_argument('--load_8bit', help='for open source models-if the model to be loaded in 8 bit', action='store_true', required=False)
parser.add_argument('--dataset', help='path to harmful questions (json) for evaluation, to be used with prompt templates for red-teaming', required=True, type=str)

args = parser.parse_args()

dataset = args.dataset
model_name = args.model
save_path = args.save_path
load_in_8bit = args.load_8bit
num_samples = args.num_samples
clean_thoughts = args.clean_thoughts
prompt = args.prompt


print(f"\n\nconfiguration")
print(f"*{'-'*10}*")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

print(f"*{'-'*10}*\n\n")


##setting up model##
if 'gpt' in model_name:

    import openai

    try:
        # API setting constants
        API_MAX_RETRY = 5
        API_RETRY_SLEEP = 10
        API_ERROR_OUTPUT = "$ERROR$"

        key_path = f'api_keys/{model_name}_api_key.json'
        with open(key_path, 'r') as f:
            keys = json.load(f)   

        openai.api_type = keys['api_type']
        openai.api_base = keys['api_base']
        openai.api_version = keys['api_version']
        openai.api_key=keys['api_key']
        model_engine = keys['model_engine']
        model_family = keys['model_family']
    except:
        raise Exception(f"\n\n\t\t\t[Sorry, please verify API key provided for {model_name} at {key_path}]")

else:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)
    if load_in_8bit:
        print("\n\n***loading model in 8 bits***\n\n")
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=load_in_8bit)



##define chat completion function##
def chat_completion(system, prompt):
    for _ in range(API_MAX_RETRY):
        try:    
            response = openai.ChatCompletion.create(
              model=model_family,
              engine=model_engine,
              messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
            response = response['choices'][0]['message']['content']
            return response
        except Exception as e:
            if 'policy' in str(e):
                print("Skipping due to openai policy")
                #such cases are tagged with [[N]]
                return '[[N]]'
            print(type(e), e)
            print("trying again")
            time.sleep(API_RETRY_SLEEP)




##process data##
def clean_thoughts_(response):

    if "(Internal thought:" in response:
        if ')' in response:
            ind =  response.index(')')+1
        else:
            ind = -1
        nresponse = response[ind:].strip()
        return nresponse

    return response


def get_context(file_name):
    f = open(file_name, "r")
    f = f.read()
    return f

def gen_prompt(q, ctx):
    prompt = ctx.replace('<question>', q.strip())
    return prompt

def process_data(dataset, ctx, nsamples):
    f = open(dataset)

    if num_samples == -1:
        data = json.load(f)
    else:
        data = json.load(f)[:nsamples]

    if dataset == 'harmfulqa.json':
        topics = []
        subtopics = []
        prompt_que = []
        orig_que = []
        for topic in data.keys():
            for subtopic in data[topic].keys():
                for q in data[topic][subtopic]:
                    orig_que.append(q)
                    prompt_que.append(gen_prompt(q))
                    topics.append(t)
                    subtopics.append(st)
    else:
        prompt_que = [gen_prompt(q, ctx) for q in data]
        orig_que = data

    return prompt_que, orig_que

context = get_context(args.prompt)
prompt_que, orig_que = process_data(dataset, context, num_samples)


##generate responses##
if not os.path.exists(save_path):
    os.makedirs(save_path)

#save file name
if clean_thoughts:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}_{model_name.split("/")[-1]}_{prompt.split("/")[-1].replace(".txt","")}_clean.json'
else:
    save_name = f'{save_path}/{dataset.split("/")[-1].replace(".json","")}_{model_name.split("/")[-1]}_{prompt.split("/")[-1].replace(".txt","")}.json'


outputs = []
system_message = ''


print("generating responses...\n")
for i in tqdm(range(len(prompt_que))):

    inputs = prompt_que[i]

    if 'gpt' in model_name:
        response = chat_completion(system=system_message, prompt=inputs)
    else:
        inputs = tokenizer([prompt_que[i]], return_tensors="pt", truncation=True, padding=True).to("cuda")
        generated_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=500)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if clean_thoughts:
        response = clean_thoughts_(response)

    question = orig_que[i]
    question2 = prompt_que[i]
    #
    if dataset == 'harmfulqa.json':
        response = [{'prompt':question, 'response':response.replace(question2,"").strip(), 'topic':topics[i], 'subtopic': subtopics[i]}]
    else:
        response = [{'prompt':question, 'response':response.replace(question2,"").strip()}]

    outputs += response

    with open(f'{save_name}', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

print(f"\nCompleted, pelase check {save_name}")


'''
How to run?
    closed source:
        python generate_responses.py --model 'chatgpt' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --num_samples 10
        python generate_responses.py --model 'chatgpt' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --num_samples 10 --clean_thoughts

        python generate_responses.py --model 'gpt4' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --num_samples 10
        python generate_responses.py --model 'gpt4' --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --num_samples 10 --clean_thoughts

    open source models:
        python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --num_samples 10
        python generate_responses.py --model lmsys/vicuna-7b-v1.3 --prompt 'red_prompts/cou.txt' --dataset hamrful_questions/dangerousqa.json --num_samples 10 --clean_thoughts
'''
