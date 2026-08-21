import json
import os
import time
import torch
import numpy as np
import shortuuid

from datasets import load_dataset
from tqdm import tqdm
from evaluation.gsm8k.utils import build_prompt, clean_answer, is_correct

def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        max_length,
        num_choices,
        teminators,
        **kwargs,
):
    questions = load_dataset(question_file)
    if questions.get('test', None) is not None:
        questions = questions['test']
    else:
        questions = questions['train']

    # Split the question file into `num_gpus` files
    # assert num_gpus_total % num_gpus_per_model == 0
    # use_ray = num_gpus_total // num_gpus_per_model > 1

    # if use_ray:
    #     import ray
    #     ray.init()
    #     get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
    #         get_model_answers
    #     ).remote
    # else:
    get_answers_func = get_model_answers

    # chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    # ans_handles = []
    # for i in range(0, len(questions), chunk_size):
        # ans_handles.append(
    get_answers_func(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        max_length,
        num_choices,
        teminators,
        **kwargs,
    )
        # )

    # if use_ray:
    #     ray.get(ans_handles)




@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        max_length,
        num_choices,
        teminators,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]
    converse_template = kwargs.pop('chat_template', 'llama-3')


    # warmup
    for wm_i in range(3):
        torch.manual_seed(0)
        
        turns = []
        steps = []
        new_tokens = []
        wall_time = []

        qs = question["question"]
        inputs = tokenizer([build_prompt(qs)], add_special_tokens=False, return_tensors="pt").to("cuda")
        
        # try:
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, step, accept_length_tree = forward_func(
            inputs,
            model,
            tokenizer,
            max_new_tokens,
            max_length,
            teminators,
            **kwargs,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        # be consistent with the template's stop_token_ids
        if teminators:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in teminators
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        

        turns.append(output)
        steps.append(int(step))
        new_tokens.append(int(new_token))
        wall_time.append(total_time)
        
        print(f"warmup {wm_i} done")

            
    print('Warmup done')

    accept_lengths_tree = []
    raw_answer = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            cur_accept_lengths_tree = []
            torch.manual_seed(i)
            
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            generate_speed = []

            qs = question["question"]
            inputs = tokenizer([build_prompt(qs)], add_special_tokens=False, return_tensors="pt").to("cuda")
            # try:
            torch.cuda.synchronize()
            start_time = time.time()
            output_ids, new_token, step, accept_length_tree = forward_func(
                inputs,
                model,
                tokenizer,
                max_new_tokens,
                max_length,
                teminators,
                **kwargs,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            accept_lengths_tree.extend(accept_length_tree)

            if teminators:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in teminators
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            turns.append(output)
            steps.append(int(step))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            generate_speed.append(int(new_token) / total_time)
            cur_accept_lengths_tree.extend(accept_length_tree)

            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree, "generate_speed": generate_speed})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                # "question_id": question["task_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

        raw_answer.append(choices[0]['turns'][0])


    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))

    result = []
    for ra, question in zip(raw_answer, questions):
        pred_ans = clean_answer(ra)
        is_cor = is_correct(pred_ans, question['answer'])
        result.append(is_cor)
    
    print('Accuracy:', sum(result) / len(result))

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

