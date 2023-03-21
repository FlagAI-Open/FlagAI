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
                 epochs=1,
                 batch_size=1,
                 lr=1e-5,
                 seed=1234,

                 fp16=False,
                 pytorch_device="cpu",
                 clip_grad=1.0,
                 checkpoint_activations=False,
                 gradient_accumulation_steps=1,
                 weight_decay=1e-5,
                 warm_up=0.1,

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
                 ):

        self.parser = argparse.ArgumentParser(description='Env args parser')
        self.parser.add_argument('--env_type', default=env_type, help='the model will be trained')
        self.parser.add_argument('--experiment_name', default=experiment_name, help='start training from saved checkpoint')
        self.parser.add_argument('--epochs', default=epochs, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--batch_size', default=batch_size, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--lr', default=lr, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--seed', default=seed, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--fp16', default=fp16, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--pytorch_device', default=pytorch_device, help='start training from saved checkpoint')
        self.parser.add_argument('--clip_grad', default=clip_grad, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--checkpoint_activations', default=checkpoint_activations, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--gradient_accumulation_steps', default=gradient_accumulation_steps, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--weight_decay', default=weight_decay, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--warm_up', default=warm_up, type=float, help='start training from saved checkpoint')
        self.parser.add_argument('--log_interval', default=log_interval, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--eval_interval', default=eval_interval, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--save_interval', default=save_interval, type=int, help='start training from saved checkpoint')
        self.parser.add_argument('--save_dir', default=save_dir, help='start training from saved checkpoint')
        self.parser.add_argument('--load_dir', default=load_dir, help='start training from saved checkpoint')
        self.parser.add_argument('--save_optim', default=save_optim, type=str2bool, help='start training from saved checkpoint')
        self.parser.add_argument('--save_rng', default=save_rng, type=str2bool,help='start training from saved checkpoint')
        self.parser.add_argument('--load_type', default=load_type, type=str2bool,help='start training from saved checkpoint')
        self.parser.add_argument('--load_optim', default=load_optim, type=str2bool,help='start training from saved checkpoint')
        self.parser.add_argument('--load_rng', default=load_rng, type=str2bool, help='start training from saved checkpoint')
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

