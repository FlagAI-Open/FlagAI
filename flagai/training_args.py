"""
Training Arguments Configuration Module
Compatible with latest version and best practices

This module provides TrainingArgs class for managing training configuration.
It supports command-line argument parsing and provides a unified interface
for training hyperparameters and distributed training settings.
"""
import argparse
from typing import Optional, List, Union


def save_best(best_score, eval_dict):
    """
    Save best model based on evaluation dictionary.
    
    Args:
        best_score: Current best score
        eval_dict: Evaluation dictionary containing 'loss' key
        
    Returns:
        Updated best score
    """
    return best_score if best_score < eval_dict['loss'] else eval_dict['loss']


def str2bool(v: Union[str, bool]) -> bool:
    """
    Convert string to boolean value.
    Compatible with argparse boolean arguments.
    
    Args:
        v: String or boolean value
        
    Returns:
        Boolean value
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ('true', '1', 'yes', 'on'):
            return True
        elif v in ('false', '0', 'no', 'off'):
            return False
    raise ValueError(f"Cannot convert {v} to boolean")


class TrainingArgs:
    """
    Training Arguments Configuration Class
    
    This class provides a unified interface for managing training configuration,
    including hyperparameters, distributed training settings, and command-line
    argument parsing.
    
    Compatible with:
    - PyTorch 2.0+
    - DeepSpeed latest version
    - BMTrain
    - Multiple training backends (pytorch, pytorchDDP, deepspeed, deepspeed+mpu, bmtrain)
    
    Usage:
        >>> from flagai.training_args import TrainingArgs
        >>> args = TrainingArgs(
        ...     env_type="pytorch",
        ...     experiment_name="my_experiment",
        ...     batch_size=64,
        ...     lr=1e-4,
        ...     epochs=10
        ... )
        >>> args = args.parse_args()  # Parse command-line arguments
    """
    
    def __init__(
        self,
        env_type: str = "pytorch",
        experiment_name: str = "test_experiment",
        model_name: str = "test_model",
        epochs: int = 1,
        batch_size: int = 1,
        lr: float = 1e-5,
        warmup_start_lr: float = 0.0,
        seed: int = 1234,
        
        # Mixed precision and device settings
        fp16: bool = False,
        pytorch_device: str = "cpu",
        clip_grad: float = 1.0,
        checkpoint_activations: bool = False,
        gradient_accumulation_steps: int = 1,
        
        # Optimizer settings
        weight_decay: float = 1e-5,
        eps: float = 1e-8,
        warm_up: float = 0.1,
        warm_up_iters: int = 0,
        skip_iters: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        
        # Logging and checkpointing
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_optim: bool = False,
        save_rng: bool = False,
        load_type: str = 'latest',  # latest, best
        load_optim: bool = False,
        load_rng: bool = False,
        tensorboard_dir: str = "tensorboard_summary",
        tensorboard: bool = False,
        
        # Weights & Biases settings
        wandb: bool = True,
        wandb_dir: str = './wandb',
        wandb_key: str = '3e614eb678063929b16c9b9aec557e2949d5a814',
        
        # Model settings
        already_fp16: bool = False,
        resume_dataset: bool = False,
        shuffle_dataset: bool = True,
        
        # Distributed training settings
        deepspeed_activation_checkpointing: bool = False,
        num_checkpoints: int = 1,
        master_ip: str = 'localhost',
        master_port: int = 17750,
        num_nodes: int = 1,
        num_gpus: int = 1,
        hostfile: str = "./hostfile",
        deepspeed_config: str = "./deepspeed.json",
        model_parallel_size: int = 1,
        training_script: str = "train.py",
        
        # LoRA settings
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        
        # BMTrain settings
        yaml_config: Optional[str] = None,
        bmt_cpu_offload: bool = True,
        bmt_lr_decay_style: str = 'cosine',
        bmt_loss_scale: float = 1024.,
        bmt_loss_scale_steps: int = 1024,
        
        # Debug and advanced settings
        bmt_async_load: bool = False,
        bmt_pre_load: bool = False,
        pre_load_dir: Optional[str] = None,
        enable_sft_dataset_dir: Optional[str] = None,
        enable_sft_dataset_file: Optional[str] = None,
        enable_sft_dataset_val_file: Optional[str] = None,
        enable_sft_dataset: bool = False,
        enable_sft_dataset_text: bool = False,
        enable_sft_dataset_jsonl: bool = False,
        enable_sft_conversations_dataset: bool = False,
        enable_sft_conversations_dataset_v2: bool = False,
        enable_sft_conversations_dataset_v3: bool = False,
        enable_weighted_dataset_v2: bool = False,
        enable_flash_attn_models: bool = False,
        IGNORE_INDEX: int = -100,
    ):
        """
        Initialize TrainingArgs with default values.
        
        Args:
            env_type: Training environment type (pytorch, pytorchDDP, deepspeed, deepspeed+mpu, bmtrain)
            experiment_name: Name of the experiment
            model_name: Name of the model
            epochs: Number of training epochs
            batch_size: Batch size per GPU
            lr: Learning rate
            warmup_start_lr: Starting learning rate for warmup
            seed: Random seed for reproducibility
            fp16: Enable mixed precision training
            pytorch_device: PyTorch device (cpu, cuda, cuda:0, etc.)
            clip_grad: Gradient clipping value
            checkpoint_activations: Enable activation checkpointing
            gradient_accumulation_steps: Number of gradient accumulation steps
            weight_decay: Weight decay coefficient
            eps: Epsilon value for optimizer
            warm_up: Warmup ratio
            warm_up_iters: Number of warmup iterations
            skip_iters: Number of iterations to skip
            adam_beta1: Adam optimizer beta1 parameter
            adam_beta2: Adam optimizer beta2 parameter
            log_interval: Logging interval
            eval_interval: Evaluation interval
            save_interval: Checkpoint saving interval
            save_dir: Directory to save checkpoints
            load_dir: Directory to load checkpoints from
            save_optim: Save optimizer state
            save_rng: Save random number generator state
            load_type: Type of checkpoint to load (latest, best)
            load_optim: Load optimizer state
            load_rng: Load random number generator state
            tensorboard_dir: TensorBoard log directory
            tensorboard: Enable TensorBoard logging
            wandb: Enable Weights & Biases logging
            wandb_dir: Weights & Biases directory
            wandb_key: Weights & Biases API key
            already_fp16: Model is already in FP16 format
            resume_dataset: Resume dataset from checkpoint
            shuffle_dataset: Shuffle dataset
            deepspeed_activation_checkpointing: Enable DeepSpeed activation checkpointing
            num_checkpoints: Number of activation checkpoints
            master_ip: Master IP address for distributed training
            master_port: Master port for distributed training
            num_nodes: Number of nodes for distributed training
            num_gpus: Number of GPUs per node
            hostfile: Hostfile for distributed training
            deepspeed_config: DeepSpeed configuration file path
            model_parallel_size: Model parallel size
            training_script: Training script path
            lora: Enable LoRA fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            lora_target_modules: List of target modules for LoRA
            yaml_config: YAML configuration file path
            bmt_cpu_offload: Enable CPU offload for BMTrain
            bmt_lr_decay_style: Learning rate decay style for BMTrain
            bmt_loss_scale: Loss scale for BMTrain
            bmt_loss_scale_steps: Loss scale steps for BMTrain
            bmt_async_load: Enable async loading for BMTrain
            bmt_pre_load: Enable pre-loading for BMTrain
            pre_load_dir: Pre-load directory
            enable_sft_dataset_dir: SFT dataset directory
            enable_sft_dataset_file: SFT dataset file
            enable_sft_dataset_val_file: SFT validation dataset file
            enable_sft_dataset: Enable SFT dataset
            enable_sft_dataset_text: Enable text SFT dataset
            enable_sft_dataset_jsonl: Enable JSONL SFT dataset
            enable_sft_conversations_dataset: Enable conversations SFT dataset
            enable_sft_conversations_dataset_v2: Enable conversations SFT dataset v2
            enable_sft_conversations_dataset_v3: Enable conversations SFT dataset v3
            enable_weighted_dataset_v2: Enable weighted dataset v2
            enable_flash_attn_models: Enable Flash Attention models
            IGNORE_INDEX: Index to ignore in loss calculation
        """
        # Set default for lora_target_modules
        if lora_target_modules is None:
            lora_target_modules = ["wq", "wv"]
        
        # Initialize argument parser
        self.parser = argparse.ArgumentParser(
            description='Training arguments parser',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Core training arguments
        self.parser.add_argument('--env_type', default=env_type, 
                                help='Training environment type (pytorch, pytorchDDP, deepspeed, deepspeed+mpu, bmtrain)')
        self.parser.add_argument('--experiment_name', default=experiment_name, 
                                help='Experiment name for logging and checkpointing')
        self.parser.add_argument('--model_name', default=model_name, 
                                help='Model name')
        self.parser.add_argument('--epochs', default=epochs, type=int, 
                                help='Number of training epochs')
        self.parser.add_argument('--batch_size', default=batch_size, type=int, 
                                help='Batch size per GPU')
        self.parser.add_argument('--lr', default=lr, type=float, 
                                help='Learning rate')
        self.parser.add_argument('--warmup_start_lr', default=warmup_start_lr, type=float, 
                                help='Starting learning rate for warmup')
        self.parser.add_argument('--seed', default=seed, type=int, 
                                help='Random seed for reproducibility')
        
        # Mixed precision and device settings
        self.parser.add_argument('--fp16', default=fp16, type=str2bool, 
                                help='Enable mixed precision training (FP16)')
        self.parser.add_argument('--pytorch_device', default=pytorch_device, 
                                help='PyTorch device (cpu, cuda, cuda:0, etc.)')
        self.parser.add_argument('--clip_grad', default=clip_grad, type=float, 
                                help='Gradient clipping value')
        self.parser.add_argument('--checkpoint_activations', default=checkpoint_activations, type=str2bool, 
                                help='Enable activation checkpointing')
        self.parser.add_argument('--gradient_accumulation_steps', default=gradient_accumulation_steps, type=int, 
                                help='Number of gradient accumulation steps')
        
        # Optimizer settings
        self.parser.add_argument('--weight_decay', default=weight_decay, type=float, 
                                help='Weight decay coefficient')
        self.parser.add_argument('--eps', default=eps, type=float, 
                                help='Epsilon value for optimizer')
        self.parser.add_argument('--warm_up', default=warm_up, type=float, 
                                help='Warmup ratio')
        self.parser.add_argument('--warm_up_iters', default=warm_up_iters, type=int, 
                                help='Number of warmup iterations')
        self.parser.add_argument('--skip_iters', default=skip_iters, type=int, 
                                help='Number of iterations to skip')
        self.parser.add_argument('--adam_beta1', default=adam_beta1, type=float, 
                                help='Adam optimizer beta1 parameter')
        self.parser.add_argument('--adam_beta2', default=adam_beta2, type=float, 
                                help='Adam optimizer beta2 parameter')
        
        # Logging and checkpointing
        self.parser.add_argument('--log_interval', default=log_interval, type=int, 
                                help='Logging interval')
        self.parser.add_argument('--eval_interval', default=eval_interval, type=int, 
                                help='Evaluation interval')
        self.parser.add_argument('--save_interval', default=save_interval, type=int, 
                                help='Checkpoint saving interval')
        self.parser.add_argument('--save_dir', default=save_dir, 
                                help='Directory to save checkpoints')
        self.parser.add_argument('--load_dir', default=load_dir, 
                                help='Directory to load checkpoints from')
        self.parser.add_argument('--save_optim', default=save_optim, type=str2bool, 
                                help='Save optimizer state')
        self.parser.add_argument('--save_rng', default=save_rng, type=str2bool, 
                                help='Save random number generator state')
        self.parser.add_argument('--load_type', default=load_type, type=str, 
                                help='Type of checkpoint to load (latest, best)')
        self.parser.add_argument('--load_optim', default=load_optim, type=str2bool, 
                                help='Load optimizer state')
        self.parser.add_argument('--load_rng', default=load_rng, type=str2bool, 
                                help='Load random number generator state')
        self.parser.add_argument('--tensorboard', default=tensorboard, type=str2bool, 
                                help='Enable TensorBoard logging')
        self.parser.add_argument('--tensorboard_dir', default=tensorboard_dir, 
                                help='TensorBoard log directory')
        
        # Weights & Biases settings
        self.parser.add_argument('--wandb', default=wandb, type=str2bool, 
                                help='Enable Weights & Biases logging')
        self.parser.add_argument('--wandb_dir', default=wandb_dir, type=str, 
                                help='Weights & Biases directory')
        self.parser.add_argument('--wandb_key', default=wandb_key, type=str, 
                                help='Weights & Biases API key')
        
        # Model settings
        self.parser.add_argument('--already_fp16', default=already_fp16, type=str2bool, 
                                help='Model is already in FP16 format')
        self.parser.add_argument('--resume_dataset', default=resume_dataset, type=str2bool, 
                                help='Resume dataset from checkpoint')
        self.parser.add_argument('--shuffle_dataset', default=shuffle_dataset, type=str2bool, 
                                help='Shuffle dataset')
        
        # Distributed training settings
        self.parser.add_argument('--deepspeed_activation_checkpointing', 
                                default=deepspeed_activation_checkpointing, type=str2bool,
                                help='Enable DeepSpeed activation checkpointing')
        self.parser.add_argument('--num_checkpoints', default=num_checkpoints, type=int, 
                                help='Number of activation checkpoints')
        self.parser.add_argument('--deepspeed_config', default=deepspeed_config, 
                                help='DeepSpeed configuration file path')
        self.parser.add_argument('--model_parallel_size', default=model_parallel_size, type=int, 
                                help='Model parallel size')
        self.parser.add_argument('--training_script', default=training_script, 
                                help='Training script path')
        self.parser.add_argument('--hostfile', default=hostfile, 
                                help='Hostfile for distributed training')
        self.parser.add_argument('--master_ip', default=master_ip, 
                                help='Master IP address for distributed training')
        self.parser.add_argument('--master_port', default=master_port, type=int, 
                                help='Master port for distributed training')
        self.parser.add_argument('--num_nodes', default=num_nodes, type=int, 
                                help='Number of nodes for distributed training')
        self.parser.add_argument('--num_gpus', default=num_gpus, type=int, 
                                help='Number of GPUs per node')
        self.parser.add_argument('--not_call_launch', action="store_true", 
                                help='Do not call launch function (for manual distributed setup)')
        self.parser.add_argument('--local-rank', default=0, type=int, 
                                help='Local rank for distributed training')
        
        # LoRA settings
        self.parser.add_argument('--lora', default=lora, type=str2bool, 
                                help='Enable LoRA fine-tuning')
        self.parser.add_argument('--lora_r', default=lora_r, type=int, 
                                help='LoRA rank')
        self.parser.add_argument('--lora_alpha', default=lora_alpha, type=float, 
                                help='LoRA alpha parameter')
        self.parser.add_argument('--lora_dropout', default=lora_dropout, type=float, 
                                help='LoRA dropout rate')
        self.parser.add_argument('--lora_target_modules', default=lora_target_modules, 
                                help='List of target modules for LoRA (comma-separated or list format)')
        
        # BMTrain settings
        self.parser.add_argument("--yaml_config", default=yaml_config, type=str, 
                                help="YAML configuration file path")
        self.parser.add_argument('--bmt_cpu_offload', default=bmt_cpu_offload, type=str2bool, 
                                help='Enable CPU offload for BMTrain')
        self.parser.add_argument('--bmt_lr_decay_style', default=bmt_lr_decay_style, type=str, 
                                help='Learning rate decay style for BMTrain')
        self.parser.add_argument('--bmt_loss_scale', default=bmt_loss_scale, type=float, 
                                help='Loss scale for BMTrain')
        self.parser.add_argument('--bmt_loss_scale_steps', default=bmt_loss_scale_steps, type=int, 
                                help='Loss scale steps for BMTrain')
        
        # Debug and advanced settings
        self.parser.add_argument('--bmt_async_load', default=bmt_async_load, type=str2bool, 
                                help='Enable async loading for BMTrain')
        self.parser.add_argument('--bmt_pre_load', default=bmt_pre_load, type=str2bool, 
                                help='Enable pre-loading for BMTrain')
        self.parser.add_argument('--pre_load_dir', default=pre_load_dir, 
                                help='Pre-load directory')
        self.parser.add_argument('--enable_sft_dataset_dir', default=enable_sft_dataset_dir, type=str, 
                                help='SFT dataset directory')
        self.parser.add_argument('--enable_sft_dataset_file', default=enable_sft_dataset_file, type=str, 
                                help='SFT dataset file')
        self.parser.add_argument('--enable_sft_dataset_val_file', default=enable_sft_dataset_val_file, type=str, 
                                help='SFT validation dataset file')
        self.parser.add_argument('--enable_sft_dataset', default=enable_sft_dataset, type=str2bool, 
                                help='Enable SFT dataset')
        self.parser.add_argument('--enable_sft_dataset_text', default=enable_sft_dataset_text, type=str2bool, 
                                help='Enable text SFT dataset')
        self.parser.add_argument('--enable_sft_dataset_jsonl', default=enable_sft_dataset_jsonl, type=str2bool, 
                                help='Enable JSONL SFT dataset')
        self.parser.add_argument('--enable_sft_conversations_dataset', default=enable_sft_conversations_dataset, type=str2bool, 
                                help='Enable conversations SFT dataset')
        self.parser.add_argument('--enable_sft_conversations_dataset_v2', default=enable_sft_conversations_dataset_v2, type=str2bool, 
                                help='Enable conversations SFT dataset v2')
        self.parser.add_argument('--enable_sft_conversations_dataset_v3', default=enable_sft_conversations_dataset_v3, type=str2bool, 
                                help='Enable conversations SFT dataset v3')
        self.parser.add_argument('--enable_weighted_dataset_v2', default=enable_weighted_dataset_v2, type=str2bool, 
                                help='Enable weighted dataset v2')
        self.parser.add_argument('--IGNORE_INDEX', default=IGNORE_INDEX, type=int, 
                                help='Index to ignore in loss calculation')
        self.parser.add_argument('--enable_flash_attn_models', default=enable_flash_attn_models, type=str2bool, 
                                help='Enable Flash Attention models')

    def add_arg(self, arg_name: str, default=None, type=str, help: str = "", store_true: bool = False):
        """
        Add a custom argument to the parser.
        
        Args:
            arg_name: Name of the argument (without -- prefix)
            default: Default value for the argument
            type: Type of the argument (str, int, float, bool, etc.)
            help: Help text for the argument
            store_true: If True, use action="store_true" for boolean flags
        """
        if store_true:
            self.parser.add_argument(f"--{arg_name}", action="store_true", help=help)
        else:
            self.parser.add_argument(f"--{arg_name}", default=default, type=type, help=help)

    def parse_args(self):
        """
        Parse command-line arguments and return parsed arguments.
        
        Handles special cases:
        - Sets not_call_launch=True for pytorch env_type
        - Converts string-formatted lists back to Python list objects
        
        Returns:
            Parsed arguments namespace
        """
        args = self.parser.parse_args()
        
        # Set not_call_launch for pytorch env_type
        if args.env_type == "pytorch":
            args.not_call_launch = True
        
        # Print parsed arguments for debugging
        print("Parsed training arguments:")
        print(args)
        
        # Convert string-formatted lists back to Python list objects
        for arg in vars(args):
            value = getattr(args, arg)
            if isinstance(value, str):
                # Remove quotes
                value = value.strip("'\"")
                # Check if it's a list format
                if value and value[0] == '[' and value[-1] == ']':
                    # Parse list string
                    value = value.strip("[] ").replace(" ", "")
                    if value:  # Non-empty list
                        value = value.split(",")
                    else:  # Empty list
                        value = []
                    setattr(args, arg, value)
        
        return args

