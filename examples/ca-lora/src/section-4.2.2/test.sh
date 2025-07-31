CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type none
CUDA_VISIBLE_DEVICES=1 python3 test_humaneval.py --lora_type q
CHECKPOINT_VERSION=checkpoint-115 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type lora
CHECKPOINT_VERSION=checkpoint-231 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type lora
CHECKPOINT_VERSION=checkpoint-346 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type lora
CHECKPOINT_VERSION=checkpoint-462 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type lora
CHECKPOINT_VERSION=checkpoint-577 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type lora
CHECKPOINT_VERSION=checkpoint-115 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type qlora
CHECKPOINT_VERSION=checkpoint-231 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type qlora
CHECKPOINT_VERSION=checkpoint-346 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type qlora
CHECKPOINT_VERSION=checkpoint-462 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type qlora
CHECKPOINT_VERSION=checkpoint-577 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type qlora
CHECKPOINT_VERSION=checkpoint-115 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type calora
CHECKPOINT_VERSION=checkpoint-231 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type calora
CHECKPOINT_VERSION=checkpoint-346 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type calora
CHECKPOINT_VERSION=checkpoint-462 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type calora
CHECKPOINT_VERSION=checkpoint-577 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type calora
CHECKPOINT_VERSION=checkpoint-346 CUDA_VISIBLE_DEVICES=0 python3 test_humaneval.py --lora_type calora-inherit-only
