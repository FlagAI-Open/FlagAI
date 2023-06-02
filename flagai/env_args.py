import argparse

def save_best(best_score, eval_dict):
    return best_score if best_score < eval_dict['loss'] else eval_dict['loss']

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

class EnvArgs:
    def __init__(self,
                 env_type="pytorch",
                 experiment_name="test_experiment",
                 model_name="test_model",
                 epochs=1,
                 batch_size=1,
                 lr=1e-5,
                 warmup_start_lr=0.0,
                 seed=1234,

                 fp16=False,
                 pytorch_device="cpu",
                 clip_grad=1.0,
                 checkpoint_activations=False,
                 gradient_accumulation_steps=1,

                 weight_decay=1e-5,
                 eps=1e-8,
                 warm_up=0.1,
                 warm_up_iters=0,
                 skip_iters=0,

                 log_interval=100,
                 eval_interval=1000,
                 save_interval=1000,

                 save_dir=None,
                 load_dir=None,
                 save_optim=False,  # save current optimizer.')
                 save_rng=False,  # save current rng state.')
                 load_type='latest',  # latest, best
                 load_optim=False,  # not load optimizer when loading checkpoint.')
                 load_rng=False,
                 tensorboard_dir="tensorboard_summary",
                 tensorboard=False,
                 wandb=True,
                 wandb_dir='./wandb',
                 wandb_key='3e614eb678063929b16c9b9aec557e2949d5a814',
                 already_fp16=False,
                 resume_dataset=False,
                 shuffle_dataset=True,

                 # distribute settings
                 deepspeed_activation_checkpointing=False,
                 num_checkpoints=1,
                 master_ip='localhost',
                 master_port=17750,
                 num_nodes=1,
                 num_gpus=1,
                 hostfile="./hostfile",
                 deepspeed_config="./deepspeed.json",
                 model_parallel_size=1,
                 training_script="train.py",

                 ## TODO optim
                 adam_beta1=0.9,
                 adam_beta2=0.999,

                 yaml_config=None,
                 bmt_cpu_offload=True,
                 bmt_lr_decay_style='cosine',
                 bmt_loss_scale=1024.,
                 bmt_loss_scale_steps=1024,

                 ## EnvTrainer Debug Only Flags
                 bmt_async_load=False,
                 bmt_pre_load=False,
                 pre_load_dir=None,
                 enable_sft_dataset_dir=None,
                 enable_sft_dataset_file=None,
                 enable_sft_dataset_val_file=None,
                 enable_sft_dataset=False,
                 enable_sft_dataset_text=False,
                 enable_sft_dataset_jsonl=False,
                 enable_sft_conversations_dataset=False,
                 enable_sft_conversations_dataset_v2=False,
                 enable_sft_conversations_dataset_v3=False,
                 enable_weighted_dataset_v2=False,

                 enable_flash_attn_models=False,
                 ):

        self.parser = argparse.ArgumentParser(description='Env args parser')
        self.parser.add_argument('--env_type', default=env_type, help='the model will be trained')
        self.parser.add_argument('--experiment_name', default=experiment_name, help='start training from saved checkpoint')
        self.parser.add_argument('--model_name', default=model_name, help='start training from saved checkpoint')
        self.parser.add_argument('--epochs', default=epochs, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--batch_size', default=batch_size, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--lr', default=lr, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--warmup_start_lr', default=warmup_start_lr, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--seed', default=seed, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--fp16', default=fp16, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--pytorch_device', default=pytorch_device, help='start training from saved checkpoint')
        self.parser.add_argument('--clip_grad', default=clip_grad, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--checkpoint_activations', default=checkpoint_activations, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--gradient_accumulation_steps', default=gradient_accumulation_steps, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--weight_decay', default=weight_decay, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--eps', default=eps, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--warm_up', default=warm_up, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--warm_up_iters', default=warm_up_iters, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--skip_iters', default=skip_iters, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--log_interval', default=log_interval, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--eval_interval', default=eval_interval, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--save_interval', default=save_interval, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--save_dir', default=save_dir, help='start training from saved checkpoint')
        self.parser.add_argument('--load_dir', default=load_dir, help='start training from saved checkpoint')
        self.parser.add_argument('--save_optim', default=save_optim, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--save_rng', default=save_rng, type=str2bool,help='start training from saved checkpoint')
        self.parser.add_argument('--load_type', default=load_type, type=str,help='start training from saved checkpoint')
        self.parser.add_argument('--load_optim', default=load_optim, type=str2bool,help='start training from saved checkpoint')
        self.parser.add_argument('--load_rng', default=load_rng, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--tensorboard', default=tensorboard, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--tensorboard_dir', default=tensorboard_dir, help='start training from saved checkpoint')
        self.parser.add_argument('--deepspeed_activation_checkpointing', default=deepspeed_activation_checkpointing, help='start training from saved checkpoint')
        self.parser.add_argument('--num_checkpoints', default=num_checkpoints, help='start training from saved checkpoint')
        self.parser.add_argument('--deepspeed_config', default=deepspeed_config, help='start training from saved checkpoint')
        self.parser.add_argument('--model_parallel_size', default=model_parallel_size, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--training_script', default=training_script, help='start training from saved checkpoint')

        self.parser.add_argument('--hostfile', default=hostfile, help='start training from saved checkpoint')
        self.parser.add_argument('--master_ip', default=master_ip, help='start training from saved checkpoint')
        self.parser.add_argument('--master_port', default=master_port, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--num_nodes', default=num_nodes, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--num_gpus', default=num_gpus, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--not_call_launch', action="store_true", help='start training from saved checkpoint')
        self.parser.add_argument('--local_rank', default=0, type=int, help='start training from saved checkpoint')

        self.parser.add_argument('--wandb', default=wandb, type=str2bool, help='whether to use wandb')
        self.parser.add_argument('--wandb_dir', default=wandb_dir, type=str, help='wandb directory')
        self.parser.add_argument('--wandb_key', default=wandb_key, type=str, help='wandb key')

        self.parser.add_argument('--already_fp16', default=already_fp16, type=str2bool, help='whether already_fp16')

        self.parser.add_argument('--resume_dataset', default=resume_dataset, type=str2bool, help='whether to resume dataset')
        self.parser.add_argument('--shuffle_dataset', default=shuffle_dataset, type=str2bool, help='start training from saved checkpoint')

        self.parser.add_argument('--adam_beta1', default=adam_beta1, type=float, help='adam beta1')
        self.parser.add_argument('--adam_beta2', default=adam_beta2, type=float, help='adam beta2')

        self.parser.add_argument('--bmt_cpu_offload', default=bmt_cpu_offload, type=str2bool, help='whther to enable cpu_offload in bmtrain')
        self.parser.add_argument('--bmt_lr_decay_style', default=bmt_lr_decay_style, type=str, help='lr scheduler type in bmtrain')
        self.parser.add_argument('--bmt_loss_scale', default=bmt_loss_scale, type=float, help='loss scale in bmtrain')
        self.parser.add_argument('--bmt_loss_scale_steps', default=bmt_loss_scale_steps, type=int, help='loss scale steps in bmtrain')

        ## TODO, Used in caller script, configs will be updated with yaml_config.
        self.parser.add_argument("--yaml_config", default=yaml_config, type=str, help="yaml config file")

        ## EnvTrainer Debug Only
        self.parser.add_argument('--bmt_async_load', default=bmt_async_load, type=str2bool, help='debug args')
        self.parser.add_argument('--bmt_pre_load', default=bmt_pre_load, type=str2bool, help='debug args')
        self.parser.add_argument('--pre_load_dir', default=pre_load_dir, help='start training from saved checkpoint')
        self.parser.add_argument('--enable_sft_dataset_dir', default=enable_sft_dataset_dir, type=str, help='debug args')
        self.parser.add_argument('--enable_sft_dataset_file', default=enable_sft_dataset_file, type=str, help='debug args')
        self.parser.add_argument('--enable_sft_dataset_val_file', default=enable_sft_dataset_val_file, type=str, help='debug args')
        self.parser.add_argument('--enable_sft_dataset', default=enable_sft_dataset, type=str2bool, help='debug args')
        self.parser.add_argument('--enable_sft_dataset_text', default=enable_sft_dataset_text, type=str2bool, help='debug args')
        self.parser.add_argument('--enable_sft_dataset_jsonl', default=enable_sft_dataset_jsonl, type=str2bool, help='debug args')
        self.parser.add_argument('--enable_sft_conversations_dataset', default=enable_sft_conversations_dataset, type=str2bool, help='debug args')
        self.parser.add_argument('--enable_sft_conversations_dataset_v2', default=enable_sft_conversations_dataset_v2, type=str2bool, help='debug args')
        self.parser.add_argument('--enable_sft_conversations_dataset_v3', default=enable_sft_conversations_dataset_v3, type=str2bool, help='debug args')
        self.parser.add_argument('--enable_weighted_dataset_v2', default=enable_weighted_dataset_v2, type=str2bool, help='debug args')
        self.parser.add_argument('--IGNORE_INDEX', default=-100, type=int, help='start training from saved checkpoint')

        self.parser.add_argument('--enable_flash_attn_models', default=enable_flash_attn_models, type=str2bool, help='debug args')

    def add_arg(self, arg_name, default=None, type=str, help="", store_true=False):
        if store_true:
            self.parser.add_argument(f"--{arg_name}", action="store_true", help=help)
        else :
            self.parser.add_argument(f"--{arg_name}", default=default, type=type, help=help)


    def parse_args(self):
        args = self.parser.parse_args()
        if args.env_type == "pytorch":
            # not need the "not_call_launch" parameter
            args.not_call_launch = True

        return args

