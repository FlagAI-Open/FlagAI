import collections
import copy




class BaseSearchSpace:
    def get_config(self, trail, args=None):
        return  {
                "do_train": True,
                "do_eval": True,
                "do_test": True,


                "save_total_limit": 1,
                # For glue datasets.
                "split_validation_test": True,
                "dataset_config_name": ["en"],
                "eval_dataset_config_name": ["en"],
                "test_dataset_config_name": ["en"],
                # other configurations.
                # To evaluate during training.
                "load_best_model_at_end": True,
                "metric_for_best_model": "average_metrics",
                "greater_is_better": True,
                "evaluation_strategy": "steps",
                "overwrite_output_dir": True,
                "push_to_hub": False,
                "save_strategy": "steps",
                "datasets_load_from_disk": args.datasets_load_from_disk,
                "datasets_saved_path": args.datasets_saved_path

            }



class BitFitSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        return {
            "delta_type": "bitfit",
            'learning_rate': learning_rate,
        }

class AdapterSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        # bottleneck_dim_base = trail.suggest_int("bottleneck_dim_base", 1, 3)
        # bottleneck_dim = int(2*4**(bottleneck_dim_base-1))
        bottleneck_dim = 32
        return {
            "delta_type": "adapter",
            'learning_rate': learning_rate,
            'bottleneck_dim': bottleneck_dim
        }

class SoftPromptSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        soft_token_num = 100
        return {
            "delta_type": "soft_prompt",
            'learning_rate': learning_rate,
            "soft_token_num":soft_token_num,
            "token_init": False,
        }

class FinetuneSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        return {
            "delta_type": "none",
            'learning_rate': learning_rate,
        }

class LoRASearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        lora_r = 8
        return {
                "delta_type": "lora",
                "learning_rate": learning_rate,
                "unfrozen_modules": [
                    "deltas",
                    "layer_norm",
                    "final_layer_norm"
                ],
                "lora_r": lora_r,
            }

class CompacterSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        return {
                "delta_type": "compacter",
                "learning_rate": learning_rate,
                "unfrozen_modules": [
                    "deltas",
                    "layer_norm",
                    "final_layer_norm"
                ],
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
                 }

class CompacterppSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        if args.model_name_or_path.split("/")[-1].startswith('t5'):
            modified_modules = [
                    "DenseReluDense"
                ]
        else:
            raise NotImplementedError

        return {
                "delta_type": "compacter",
                "learning_rate": learning_rate,
                "unfrozen_modules": [
                    "deltas",
                    "layer_norm",
                    "final_layer_norm"
                ],
                "modified_modules": modified_modules,
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
                 }

class LowRankAdapterSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        low_rank_rank = 1
        return  {
                    "delta_type": "low_rank_adapter",
                    "learning_rate": learning_rate,
                    "unfrozen_modules": [
                        "deltas",
                        "layer_norm",
                        "final_layer_norm"
                    ],
                    "non_linearity": "gelu_new",
                    "low_rank_w_init": "glorot-uniform",
                    "low_rank_rank": low_rank_rank,
                }



class PrefixSearchSpace:
    def get_config(self, trail, args=None):
        learning_rate = trail.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        return {
                "delta_type": "prefix",
                "learning_rate": learning_rate,
                "unfrozen_modules": [
                    "deltas",
                ],
        }




class T5BaseSearchSpace:
    def get_config(self, trail, args=None):
        batch_size_base = trail.suggest_int('batch_size_base', 1, 4)
        if batch_size_base >= 4:
            gradient_accumulation_steps = 2**(batch_size_base-3)
        else:
            gradient_accumulation_steps = 1
        batch_size =  int(16 * 2**(min(batch_size_base,3)-1))
        warmup_steps = trail.suggest_categorical('warmup_steps', [0, 500])
        return {
            "model_name_or_path": f"{args.plm_path_base}t5-base", # change here for loading from custom path
            "tokenizer_name": f"{args.plm_path_base}t5-base",   # change here for loading from custom path
            'batch_size':batch_size,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "save_steps": 200,
            "eval_steps": 200,
            "max_steps": 5000,
            "predict_with_generate": True,
        }


class RobertaBaseSearchSpace:
    def get_config(self, trail, args=None):
        batch_size_base = trail.suggest_int('batch_size_base', 1, 4)
        if batch_size_base >= 4:
            gradient_accumulation_steps = 2**(batch_size_base-3)
        else:
            gradient_accumulation_steps = 1
        batch_size =  int(16 * 2**(min(batch_size_base,3)-1))
        warmup_steps = trail.suggest_categorical('warmup_steps', [0, 500])
        return {
            "model_name_or_path": f"{args.plm_path_base}roberta-base", # change here for loading from custom path
            "tokenizer_name": f"{args.plm_path_base}roberta-base",   # change here for loading from custom path
            'batch_size':batch_size,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "save_steps": 200,
            "eval_steps": 200,
            "max_steps": 5000,
            "predict_with_generate": False,
        }



class DatasetSearchSpace:
    dataset_order = ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record", "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte",  "mnli", "qqp", "stsb", "wnli"]
    dataset_config = {("task_name", "eval_dataset_name", "test_dataset_name",
    "max_source_length"): list(zip(
        ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record", "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte",  "mnli", "qqp", "stsb", "wnli"],
        ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record", "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte",  "mnli", "qqp", "stsb", "wnli"],
        ["superglue-boolq", "superglue-cb", "superglue-copa", "superglue-wic", "superglue-multirc", "superglue-record", "superglue-wsc.fixed", "mrpc", "cola", "sst2", "qnli", "rte", "mnli", "qqp", "stsb", "wnli"],
        [256, 256, 256, 256, 256, 512, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128],
    ))}
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        fields = list(list(self.dataset_config.keys())[0])
        dataset_id = self.dataset_order.index(dataset_name)
        values = list(self.dataset_config.values())[0][dataset_id]
        self.fixed_params =  {f:v for f, v in zip(fields, values)}

    def get_config(self, trail, args=None):
        return self.fixed_params





AllDeltaSearchSpace = {
    "none": FinetuneSearchSpace,
    "bitfit": BitFitSearchSpace,
    "adapter": AdapterSearchSpace,
    "compacter": CompacterSearchSpace,
    "compacterpp": CompacterppSearchSpace,
    "lora": LoRASearchSpace,
    "prefix": PrefixSearchSpace,
    "lowrankadapter":LowRankAdapterSearchSpace,

}

AllBackboneSearchSpace = {
    "t5-base": T5BaseSearchSpace,
    "roberta-base": RobertaBaseSearchSpace,
}

