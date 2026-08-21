import collections
import copy

PATHBASE="/mnt/sfs_turbo/hsd/plm_cache/"
PATHBASE="/home/hushengding/plm_cache/"

AllConfigs = {}

import argparse
import json
import os
parser = argparse.ArgumentParser("Parser to generate configuration")
parser.add_argument("--job", type=str)
parser.add_argument("--")
args = parser.parse_args()


if __name__ == "__main__":

    config = AllConfigs[args.job]

    Cartesian_product = []
    for key in config:
        if isinstance(key, tuple):
            Cartesian_product.append(key)
    all_config_jsons = {}
    for key_tuple in Cartesian_product:
        for zipped in config[key_tuple]:
            job_name = zipped[0]
            all_config_jsons[job_name] = {}
            for key_name, zipped_elem in zip(key_tuple, zipped):
                if key_name != 'job_name':
                    all_config_jsons[job_name][key_name] = zipped_elem
    for key in config:
        if not isinstance(key, tuple):
            for job_name in all_config_jsons:
                if key == "output_dir":
                    all_config_jsons[job_name][key] = config[key] + job_name
                else:
                    all_config_jsons[job_name][key] = config[key]


    if not os.path.exists(f"configs/{args.job}/"):
        os.mkdir(f"configs/{args.job}/")

    for job_name in all_config_jsons:
        with open(f"configs/{args.job}/{job_name}.json", 'w') as fout:
            json.dump(all_config_jsons[job_name], fout, indent=4,sort_keys=True)



