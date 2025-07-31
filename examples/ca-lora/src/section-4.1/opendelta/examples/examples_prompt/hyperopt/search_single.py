
import os
import argparse
import random
import json
from examples_prompt.search_space import AllBackboneSearchSpace, AllDeltaSearchSpace, BaseSearchSpace, DatasetSearchSpace
import optuna
from functools import partial
from optuna.samplers import TPESampler
import shutil
import time

import subprocess


def objective_singleseed(args, unicode, search_space_sample  ):
    os.mkdir(f"{args.output_dir}/{unicode}")
    search_space_sample.update({"output_dir": f"{args.output_dir}/{unicode}"})


    with open(f"{args.output_dir}/{unicode}/this_configs.json", 'w') as fout:
        json.dump(search_space_sample, fout, indent=4,sort_keys=True)


    command = "CUDA_VISIBLE_DEVICES={} ".format(args.cuda_id)
    command += f"{args.pythonpath} {args.main_file_name} "
    command += f"{args.output_dir}/{unicode}/this_configs.json"
    command += f" >> {args.output_dir}/{unicode}/output.log 2>&1"


    print("======"*5+"\n"+command)
    p = subprocess.Popen(command, cwd=f"{args.pathbase}", shell=True)
    print(f"wait for subprocess \"{command}\" to complete")
    p.wait()

    # if status_code != 0:
    #     with open(f"{args.output_dir}/{args.cuda_id}.log",'r') as flog:
    #         lastlines = " ".join(flog.readlines()[-100:])
    #         if "RuntimeError: CUDA out of memory." in lastlines:
    #             time.sleep(600)  # sleep ten minites and try again
    #             shutil.rmtree(f"{args.output_dir}/{unicode}/")
    #             return objective_singleseed(args, unicode, search_space_sample)
    #         else:
    #             raise RuntimeError("error in {}".format(unicode))



    with open(f"{args.output_dir}/{unicode}/results.json", 'r') as fret:
        results =json.load(fret)

    for filename in os.listdir(f"{args.output_dir}/{unicode}/"):
        if not filename.endswith("this_configs.json"):
            full_file_name = f"{args.output_dir}/{unicode}/{filename}"
            if os.path.isdir(full_file_name):
                shutil.rmtree(f"{args.output_dir}/{unicode}/{filename}")
            else:
                os.remove(full_file_name)

    results_all_test_datasets = []
    print("results:", results)
    for datasets in results['test']:
        results_all_test_datasets.append(results['test'][datasets]['test_average_metrics'])

    return sum(results_all_test_datasets)/len(results_all_test_datasets)#results['test']['average_metrics']



def objective(trial, args=None):
    search_space_sample = {}
    search_space_sample.update(BaseSearchSpace().get_config(trial, args))
    search_space_sample.update(AllBackboneSearchSpace[args.model_name]().get_config(trial, args))
    search_space_sample.update(DatasetSearchSpace(args.dataset).get_config(trial, args))
    search_space_sample.update(AllDeltaSearchSpace[args.delta_type]().get_config(trial, args))
    results = []
    for seed in range(42, 42+args.repeat_time):
        search_space_sample.update({"seed": seed})
        unicode = random.randint(0, 100000000)
        while os.path.exists(f"{args.output_dir}/{unicode}"):
            unicode = unicode+1
        trial.set_user_attr("trial_dir", f"{args.output_dir}/{unicode}")
        res = objective_singleseed(args, unicode = unicode, search_space_sample=search_space_sample)
        results.append(res)
    ave_res = sum(results)/len(results)
    return -ave_res




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_type")
    parser.add_argument("--dataset")
    parser.add_argument("--model_name")
    parser.add_argument("--cuda_id", type=int)
    parser.add_argument("--main_file_name", type=str)
    parser.add_argument("--study_name")
    parser.add_argument("--num_trials", type=int)
    parser.add_argument("--repeat_time", type=int)
    parser.add_argument("--optuna_seed", type=int, default="the seed to sample suggest point")
    parser.add_argument("--pathbase", type=str, default="")
    parser.add_argument("--pythonpath", type=str, default="")
    parser.add_argument("--plm_path_base", type=str, default="")
    parser.add_argument("--datasets_load_from_disk", action="store_true")
    parser.add_argument("--datasets_saved_path", type=str)

    args = parser.parse_args()


    setattr(args, "output_dir", f"{args.pathbase}/outputs_search/{args.study_name}")

    study = optuna.load_study(study_name=args.study_name, storage=f'sqlite:///{args.study_name}.db', sampler=TPESampler(seed=args.optuna_seed))
    study.optimize(partial(objective, args=args), n_trials=args.num_trials)

    print("complete single!")



