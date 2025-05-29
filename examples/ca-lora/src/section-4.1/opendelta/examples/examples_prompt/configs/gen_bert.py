import collections
import copy

PATHBASE="/mnt/sfs_turbo/hsd/plm_cache/"
# PATHBASE="/home/hushengding/plm_cache/"

AllConfigs = {}

BaseConfigs = {}


#### ROBERTA######
BaseConfigs['bert-base-cased'] = {
                ("job_name", "task_name", "eval_dataset_name", "test_dataset_name", "num_train_epochs",
                "max_source_length",
                "per_device_train_batch_size", "per_device_eval_batch_size", "warmup_steps","save_steps", "eval_steps"): zip(
                    ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record",
                    "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte",  "mnli", "qqp", "stsb"],
                    ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record", "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte",  "mnli", "qqp", "stsb"],
                    ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record", "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte",  "mnli", "qqp", "stsb"],
                    ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record", "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte", "mnli", "qqp", "stsb"],
                    [ 20,  20,  40,  20,   3,   3,  20,  20,  20,   3,   3,  20,   3,   3,  20],
                    [256, 256, 256, 256, 256, 512, 256, 128, 128, 128, 128, 128, 128, 128, 128],
                    [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    [0] *7 +[0] *8,
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                ),
                "do_train": True,
                "do_eval": True,
                "do_test": True,

                "model_name_or_path": f"{PATHBASE}bert-base-cased",
                "tokenizer_name": f"{PATHBASE}bert-base-cased",
                "save_total_limit": 1,
                # For glue datasets.
                "is_seq2seq": False,
                "split_validation_test": True,
                "seed": 42,
                "dataset_config_name": ["en"],
                "eval_dataset_config_name": ["en"],
                "test_dataset_config_name": ["en"],
                # other configurations.
                "predict_with_generate": False,
                # To evaluate during training.
                "load_best_model_at_end": True,
                "metric_for_best_model": "average_metrics",
                "greater_is_better": True,
                "evaluation_strategy": "steps",
                "overwrite_output_dir": True,
                "push_to_hub": False,
                "push_to_delta_center": True,
                "save_strategy": "steps",
                "datasets_load_from_disk": True,
                "datasets_saved_path": "/mnt/sfs_turbo/hsd/huggingface_datasets/saved_to_disk/"
            }

AllConfigs['prefix_bert-base-cased'] = copy.deepcopy(BaseConfigs['bert-base-cased'])
AllConfigs['prefix_bert-base-cased'].update({
                                "delta_type": "prefix",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                ],
                                "output_dir": "outputs/prefix/bert-base-cased/",
                            })

AllConfigs['soft_prompt_bert-base-cased'] = copy.deepcopy(BaseConfigs['bert-base-cased'])
AllConfigs['soft_prompt_bert-base-cased'].update({
                                "delta_type": "soft_prompt",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                ],
                                "output_dir": "outputs/soft_prompt/bert-base-cased/",
                            })

AllConfigs['prefix_bert-large-cased'] = copy.deepcopy(AllConfigs['prefix_bert-base-cased'])
AllConfigs['prefix_bert-large-cased'].update({
    "output_dir": "outputs/prefix/bert-large-cased/",
    "model_name_or_path": f"{PATHBASE}bert-large-cased",
    "tokenizer_name": f"{PATHBASE}bert-large-cased",
})

if __name__ == "__main__":
    import argparse
    import json
    import os
    parser = argparse.ArgumentParser("Parser to generate configuration")
    parser.add_argument("--job", type=str)
    args = parser.parse_args()

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



