import collections
import copy

PATHBASE="/mnt/sfs_turbo/hsd/plm_cache/"
PATHBASE="/home/hushengding/plm_cache/"

AllConfigs = {}

BaseConfigs = {}

#### ROBERTA ######
BaseConfigs['bigbird-roberta-large'] = {
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

                "model_name_or_path": f"{PATHBASE}bigbird-roberta-large",
                "tokenizer_name": f"{PATHBASE}bigbird-roberta-large",
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
                "push_to_hub": True,
                "save_strategy": "steps"
            }



AllConfigs['bitfit_bigbird-roberta-large'] = copy.deepcopy(BaseConfigs['bigbird-roberta-large'])
AllConfigs['bitfit_bigbird-roberta-large'].update({
                "delta_type": "bitfit",
                "learning_rate": 1e-3,
                "output_dir": "outputs/bitfit/bigbird-roberta-large/",
            })

AllConfigs['none_bigbird-roberta-large'] = copy.deepcopy(BaseConfigs['bigbird-roberta-large'])
AllConfigs['none_bigbird-roberta-large'].update({
                "delta_type": "none",
                "learning_rate": 1e-5,
                "output_dir": "outputs/none/bigbird-roberta-large/",
            })


AllConfigs['lora_bigbird-roberta-large'] = copy.deepcopy(BaseConfigs['bigbird-roberta-large'])
AllConfigs['lora_bigbird-roberta-large'].update({
                "delta_type": "lora",
                "learning_rate": 1e-3,
                "modified_modules": [
                    "query",
                    "key",
                ],
                "output_dir": "outputs/lora/bigbird-roberta-large/",
            })

AllConfigs['adapter_bigbird-roberta-large'] = copy.deepcopy(BaseConfigs['bigbird-roberta-large'])
AllConfigs['adapter_bigbird-roberta-large'].update({
                "delta_type": "adapter",
                "learning_rate": 1e-3,
                "output_dir": "outputs/adapter/bigbird-roberta-large/",
            })

AllConfigs['low_rank_adapter_bigbird-roberta-large'] = copy.deepcopy(BaseConfigs['bigbird-roberta-large'])
AllConfigs['low_rank_adapter_bigbird-roberta-large'].update({
                "delta_type": "low_rank_adapter",
                "learning_rate": 1e-3,
                "output_dir": "outputs/low_rank_adapter/bigbird-roberta-large/",
            })


AllConfigs['soft_prompt_bigbird-roberta-large'] = copy.deepcopy(BaseConfigs['bigbird-roberta-large'])
AllConfigs['soft_prompt_bigbird-roberta-large'].update({
                                "delta_type": "soft_prompt",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                ],
                                "output_dir": "outputs/soft_prompt/bigbird-roberta-large/",
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



