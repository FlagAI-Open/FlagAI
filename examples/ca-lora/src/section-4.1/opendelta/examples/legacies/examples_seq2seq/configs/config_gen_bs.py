import collections
import copy

BS = 1
AllConfigs = {}

BaseConfigs = {}
BaseConfigs['t5-base'] = {
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
                    # [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    # [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    [ 32,  32,  32,  32,  32,  16,  32] + [BS] * 8,
                    [ 32,  32,  32,  32,  32,  16,  32] + [BS] * 8,
                    [0] *7 +[0] *8,
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                ),
                "do_train": True,
                "do_eval": True,
                "do_test": True,

                "model_name_or_path": "t5-base",
                "tokenizer_name": "t5-base",
                "save_total_limit": 1,
                # For glue datasets.
                "split_validation_test": True,
                "seed": 42,
                "dataset_config_name": ["en"],
                "eval_dataset_config_name": ["en"],
                "test_dataset_config_name": ["en"],
                # other configurations.
                "predict_with_generate": True,
                # To evaluate during training.
                "load_best_model_at_end": True,
                "metric_for_best_model": "average_metrics",
                "greater_is_better": True,
                "evaluation_strategy": "steps",
                "overwrite_output_dir": True,
                "push_to_hub": True,
                "save_strategy": "steps"
            }

BaseConfigs['t5-large'] = {
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
                    # [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    # [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    [ 32,  32,  32,  32,  32,  16,  32] + [BS] * 8,
                    [ 32,  32,  32,  32,  32,  16,  32] + [BS] * 8,
                    [0] *7 +[0] *8,
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                ),
                "do_train": True,
                "do_eval": True,
                "do_test": True,

                "model_name_or_path": "/home/hushengding/plm_cache/t5-large",
                "tokenizer_name": "/home/hushengding/plm_cache/t5-large",
                "save_total_limit": 1,
                # For glue datasets.
                "split_validation_test": True,
                "seed": 42,
                "dataset_config_name": ["en"],
                "eval_dataset_config_name": ["en"],
                "test_dataset_config_name": ["en"],
                # other configurations.
                "predict_with_generate": True,
                # To evaluate during training.
                "load_best_model_at_end": True,
                "metric_for_best_model": "average_metrics",
                "greater_is_better": True,
                "evaluation_strategy": "steps",
                "overwrite_output_dir": True,
                "push_to_hub": True,
                "save_strategy": "steps"
            }

BaseConfigs['t5-3b'] = {
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
                    # [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    # [ 32,  32,  32,  32,  32,  16,  32] + [32] * 8,
                    [ 32,  32,  32,  32,  32,  16,  32] + [BS] * 8,
                    [ 32,  32,  32,  32,  32,  16,  32] + [BS] * 8,
                    [0] *7 +[0] *8,
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                    [200, 100, 50, 100, 200, 200, 100, 200, 100, 200, 200, 100, 200, 200, 100],
                ),
                "do_train": True,
                "do_eval": True,
                "do_test": True,

                "model_name_or_path": "/home/hushengding/plm_cache/t5-3b",
                "tokenizer_name": "/home/hushengding/plm_cache/t5-3b",
                "save_total_limit": 1,
                # For glue datasets.
                "split_validation_test": True,
                "seed": 42,
                "dataset_config_name": ["en"],
                "eval_dataset_config_name": ["en"],
                "test_dataset_config_name": ["en"],
                # other configurations.
                "predict_with_generate": True,
                # To evaluate during training.
                "load_best_model_at_end": True,
                "metric_for_best_model": "average_metrics",
                "greater_is_better": True,
                "evaluation_strategy": "steps",
                "overwrite_output_dir": True,
                "push_to_hub": True,
                "save_strategy": "steps"
            }

AllConfigs['bitfit_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['bitfit_t5-base'].update({
                "delta_type": "bitfit",
                "learning_rate": 3e-4,
                "output_dir": "outputs/bitfit/t5-base/",
            })



AllConfigs['adapter_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['adapter_t5-base'].update({
                                "delta_type": "adapter",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "bottleneck_dim":24,
                                "output_dir": "outputs/adapter/t5-base/",
                            })

AllConfigs['lora_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['lora_t5-base'].update({
                                "delta_type": "lora",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "lora_r": 8,
                                "output_dir": "outputs/lora/t5-base/",
                            })

AllConfigs['compacter_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['compacter_t5-base'].update({
                                "delta_type": "compacter",
                                "learning_rate": 3e-3,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "output_dir": "outputs/compacter/t5-base/",
                                "non_linearity": "gelu_new",

                                #Compacter.
                                "hypercomplex_division": 4,
                                "hypercomplex_adapters": True,
                                "hypercomplex_nonlinearity": "glorot-uniform",
                                # gradient clip and clamp
                                "gradient_clip": False,
                                "phm_clamp": False,
                                "normalize_phm_weight": False,
                                "learn_phm": True,
                                # shared one side
                                "factorized_phm": True,
                                "shared_phm_rule": False,
                                "factorized_phm_rule": False,
                                "phm_c_init": "normal",
                                "phm_init_range": 0.0001,
                                "use_bias_down_sampler": True,
                                "use_bias_up_sampler": True,
                            })

AllConfigs['compacter++_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['compacter++_t5-base'].update({
                                "delta_type": "compacter",
                                "learning_rate": 3e-3,
                                "do_train": True,
                                "do_eval": True,
                                "do_test": True,
                                "modified_modules": [
                                    "DenseReluDense"
                                ],
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "output_dir": "outputs/compacter++/t5-base/",
                                "non_linearity": "gelu_new",

                                #Compacter.
                                "hypercomplex_division": 4,
                                "hypercomplex_adapters": True,
                                "hypercomplex_nonlinearity": "glorot-uniform",
                                # gradient clip and clamp
                                "gradient_clip": False,
                                "phm_clamp": False,
                                "normalize_phm_weight": False,
                                "learn_phm": True,
                                # shared one side
                                "factorized_phm": True,
                                "shared_phm_rule": False,
                                "factorized_phm_rule": False,
                                "phm_c_init": "normal",
                                "phm_init_range": 0.0001,
                                "use_bias_down_sampler": True,
                                "use_bias_up_sampler": True,
                            })


AllConfigs['low_rank_adapter_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['low_rank_adapter_t5-base'].update({
                                "delta_type": "low_rank_adapter",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "output_dir": "outputs/low_rank_adapter/t5-base/",
                                "non_linearity": "gelu_new",
                                "low_rank_w_init": "glorot-uniform",
                                "low_rank_rank": 1,
                            })


AllConfigs['soft_prompt_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['soft_prompt_t5-base'].update({
                                "delta_type": "soft_prompt",
                                "learning_rate": 3e-2,
                                "soft_token_num":100,
                                "token_init": False,
                                "unfrozen_modules": [
                                    "deltas",
                                ],
                                "output_dir": "outputs/soft_prompt/t5-base/",
                            })

AllConfigs['prefix_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['prefix_t5-base'].update({
                                "delta_type": "prefix",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                ],
                                "output_dir": "outputs/prefix/t5-base/",
                            })

AllConfigs['none_t5-base'] = copy.deepcopy(BaseConfigs['t5-base'])
AllConfigs['none_t5-base'].update({
                                "delta_type": "none",
                                "learning_rate": 3e-5,
                                "output_dir": "outputs/none/t5-base/",
                            })

AllConfigs['bitfit_t5-large'] = copy.deepcopy(BaseConfigs['t5-large'])
AllConfigs['bitfit_t5-large'].update({
                "delta_type": "bitfit",
                "learning_rate": 3e-4,
                "output_dir": "outputs/bitfit/t5-large/",
            })

AllConfigs['none_t5-large'] = copy.deepcopy(BaseConfigs['t5-large'])
AllConfigs['none_t5-large'].update({
                                "delta_type": "none",
                                "learning_rate": 3e-5,
                                "output_dir": "outputs/none/t5-large/",
                            })


AllConfigs['bitfit_t5-3b'] = copy.deepcopy(BaseConfigs['t5-3b'])
AllConfigs['bitfit_t5-3b'].update({
                "delta_type": "bitfit",
                "learning_rate": 3e-4,
                "output_dir": "outputs/bitfit/t5-3b/",
            })

AllConfigs['none_t5-3b'] = copy.deepcopy(BaseConfigs['t5-3b'])
AllConfigs['none_t5-3b'].update({
                                "delta_type": "none",
                                "learning_rate": 3e-5,
                                "output_dir": "outputs/none/t5-3b/",
                            })

AllConfigs['adapter_t5-3b'] = copy.deepcopy(BaseConfigs['t5-3b'])
AllConfigs['adapter_t5-3b'].update({
                                "delta_type": "adapter",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "bottleneck_dim":24,
                                "output_dir": "outputs/adapter/t5-3b/",
                            })

AllConfigs['adapter_t5-large'] = copy.deepcopy(BaseConfigs['t5-large'])
AllConfigs['adapter_t5-large'].update({
                                "delta_type": "adapter",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "bottleneck_dim":24,
                                "output_dir": "outputs/adapter/t5-large/",
                            })

AllConfigs['lora_t5-large'] = copy.deepcopy(BaseConfigs['t5-large'])
AllConfigs['lora_t5-large'].update({
                                "delta_type": "lora",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "lora_r": 8,
                                "output_dir": "outputs/lora/t5-large/",
                            })

AllConfigs['lora_t5-3b'] = copy.deepcopy(BaseConfigs['t5-3b'])
AllConfigs['lora_t5-3b'].update({
                                "delta_type": "lora",
                                "learning_rate": 3e-4,
                                "unfrozen_modules": [
                                    "deltas",
                                    "layer_norm",
                                    "final_layer_norm"
                                ],
                                "lora_r": 8,
                                "output_dir": "outputs/lora/t5-3b/",
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


    if not os.path.exists(f"./{args.job}_{BS}/"):
        os.mkdir(f"./{args.job}_{BS}/")

    for job_name in all_config_jsons:
        with open(f"./{args.job}_{BS}/{job_name}.json", 'w') as fout:
            json.dump(all_config_jsons[job_name], fout, indent=4,sort_keys=True)



