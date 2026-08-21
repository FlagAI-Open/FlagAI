import json
import os
import re
import numpy as np
import torch
from flagai.auto_model.auto_loader import AutoLoader
import numpy as np
from flagai.model.predictor.aquila import aquila_generate
from tqdm import tqdm
from flagai.auto_model.auto_loader import AutoLoader

def find_answer(s):
    assert('boxed' in s)
    ans = s.split('boxed')[-1]
    if(ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if(c == '{'): 
                stack += 1
                a += c
            elif(c == '}'): 
                stack -= 1
                if(stack == 0): break
                a += c
            else: 
                a += c
    else:
        a = ans.split('$')[0].strip()
    return a

def test_answer(pred_str, ans_str):
    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    else:
        pattern = '\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else: pred = ''

    gold = find_answer(ans_str)
    # gold = re.findall(pattern, ans_str)
    # print(ans_str)
    # gold = gold[-1]
    # print('pred:', pred)
    # print('gold:', gold)
    # print('---\n\n')
    return pred == gold

def parse_pred_ans(filename):
    with open(filename) as fd: lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        if(l.startswith('Q: ')):
            if(am is not None and a is not None):
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if(test_answer(am, a)):
                    acc += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif(l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif(l.startswith('A:')):
            current_mode = 'a'
            a = l
        else:
            if(current_mode == 'q'): q += l
            elif(current_mode == 'am'): am += l
            elif(current_mode == 'a'): a += l
            else:
                raise ValueError(current_mode)
                
    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if(test_answer(am, a)):
        acc += 1
    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold

def load():
    state_dict = "./checkpoints_in"
    model_name = 'aquilachat-7b'
    device = "cuda:0"

    loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True,
    fp16=True)

    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    model.eval()
    model.half()
    model.to(device)

    return model, tokenizer

math_algebra = []
dir_name = 'math_dataset/train/algebra'
for filename in os.listdir(dir_name):
    if(filename.endswith('.json')):
        d = json.load(open(dir_name + '/' + filename))
        math_algebra.append(d)
        
# dev_idx = np.load('math_dataset/train/algebra_dev_idx.npy')
# math_algebra_dev = [math_algebra[i] for i in dev_idx]

math_algebra_dev = math_algebra

print(math_algebra_dev[0])

prompt_grade_1 = open('lib_prompt/algebra/prompt_grad_1.txt').read()
prompt_grade_3 = open('lib_prompt/algebra/prompt_grad_3.txt').read()
prompt_grade_5 = open('lib_prompt/algebra/prompt_grad_5.txt').read()

model, tokenizer = load()
## grad_1
i = 0
with open('outputs/dev_algebra_grad_1.txt', 'w') as fd:
    for d in tqdm(math_algebra_dev):
        q = d['problem']
        a = d['solution']
        prompt_q = prompt_grade_1 + '\nQuestion: ' + q + '\n'

        with torch.no_grad():
            response = aquila_generate(tokenizer, model, prompts=[prompt_q], max_gen_len=256,
                                       temperature=0.001)

        # ans_model = response['choices'][0]['text']
        fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, response, a))
        i += 1
        # if(i == 10): break
    
questions, ans_pred, ans_gold = parse_pred_ans("outputs/dev_algebra_grad_1.txt")

