"""
CLITrainer Training Script for ViT on CIFAR-100
Compatible with latest version and best practices

CLITrainer is a unified training interface that supports:
- Single GPU/CPU training (pytorch)
- Multi-GPU DDP training (pytorchDDP)
- DeepSpeed training (deepspeed)
- DeepSpeed with model parallelism (deepspeed+mpu)
- BMTrain training (bmtrain)

Usage:
    # Single GPU/CPU
    python train_cli_trainer.py --env_type=pytorch
    
    # Multi-GPU DDP
    python train_cli_trainer.py --env_type=pytorchDDP --num_gpus=8
    
    # DeepSpeed
    deepspeed --num_gpus=8 train_cli_trainer.py --env_type=deepspeed
    
    # With command line arguments
    python train_cli_trainer.py --env_type=pytorch --batch_size=128 --lr=1e-4 --epochs=100
"""
import torch
import os
from torchvision import transforms
from torchvision.datasets import CIFAR100
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from flagai.cli_trainer import CLITrainer
from flagai.auto_model.auto_loader import AutoLoader
from flagai.training_args import TrainingArgs

# Training hyperparameters
lr = 2e-5
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize TrainingArgs with default values
env_args = TrainingArgs(
    env_type="pytorch",  # Options: pytorch, pytorchDDP, deepspeed, deepspeed+mpu, bmtrain
    experiment_name="vit-cifar100-single_gpu",
    batch_size=64,
    num_gpus=1,
    gradient_accumulation_steps=1,
    lr=lr,
    weight_decay=1e-5,
    epochs=n_epochs,
    log_interval=100,
    eval_interval=1000,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_vit_cifar100_single_gpu",
    save_interval=1000,
    num_checkpoints=1,
    # Additional settings
    fp16=False,  # Enable mixed precision training if needed
    clip_grad=1.0,
    seed=42,  # For reproducibility
    tensorboard=True,  # Enable TensorBoard logging
    wandb=False,  # Enable Weights & Biases logging if needed
)

# Add custom arguments if needed
# env_args.add_arg(arg_name="custom_arg", default=0, type=int)

# Parse command line arguments (overrides defaults)
env_args = env_args.parse_args()

# Initialize CLITrainer
trainer = CLITrainer(env_args)

def build_cifar():
    """
    Build CIFAR-100 datasets with data augmentation.
    Compatible with PyTorch 2.0+ transforms.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR100(root="./data/cifar100", train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root="./data/cifar100", train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def collate_fn(batch):
    """
    Collate function for batching data.
    Compatible with PyTorch 2.0+ and supports mixed precision training.
    """
    images = torch.stack([b[0] for b in batch])
    # Handle mixed precision if enabled
    if hasattr(trainer, 'fp16') and trainer.fp16:
        images = images.half()
    labels = [b[1] for b in batch]
    labels = torch.tensor(labels, dtype=torch.long)  # Use dtype instead of .long()
    return {"images": images, "labels": labels}

def validate(logits, labels, meta=None):
    """
    Validation metric function for accuracy calculation.
    Compatible with distributed evaluation.
    """
    _, predicted = logits.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    """
    Main training loop using CLITrainer.
    Compatible with latest version and best practices.
    
    CLITrainer provides a unified interface for different training backends:
    - Automatically handles distributed setup
    - Supports multiple training frameworks (PyTorch, DeepSpeed, BMTrain)
    - Manages checkpointing, logging, and evaluation
    """
    # Set random seeds for reproducibility
    seed = getattr(env_args, 'seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load model
    loader = AutoLoader(task_name="classification",
                        model_name="vit-base-p16-224",
                        num_classes=100)

    model = loader.get_model()
    
    # Setup optimizer with PyTorch 2.0+ optimizations
    # Using fused=True for better performance (requires PyTorch 2.0+)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=env_args.lr,
        weight_decay=env_args.weight_decay,
        fused=torch.cuda.is_available() and hasattr(torch.optim.Adam, 'fused')  # PyTorch 2.0+ feature
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, env_args.epochs)
    
    # Load datasets
    train_dataset, val_dataset = build_cifar()

    # Start training with CLITrainer
    # CLITrainer automatically handles:
    # - Model preparation (pre_train)
    # - Training loop (do_train)
    # - Distributed setup
    # - Checkpointing
    # - Logging (TensorBoard, WandB)
    # - Evaluation
    trainer.train(
        model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataset=train_dataset,
        valid_dataset=val_dataset,
        metric_methods=[["accuracy", validate]],
        collate_fn=collate_fn,
        find_unused_parameters=False  # Set to False for better performance if all parameters are used
    )
    
    # Cleanup (handled automatically by CLITrainer, but good practice)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

