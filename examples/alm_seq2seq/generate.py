# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import sys 
sys.path.append("/sharefs/baai-mrnd/yzd/FlagAI")
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
import random
import numpy as np 


# def model_parallel_cuda_manual_seed(seed):
#     """Initialize model parallel cuda seed.

#     This function should be called after the model parallel is
#     initialized. Also, no torch.cuda.manual_seed should be called
#     after this function. Basically, this is replacement for that
#     function.
#     Two set of RNG states are tracked:
#         default state: This is for data parallelism and is the same among a
#                        set of model parallel GPUs but different across
#                        different model paralle groups. This is used for
#                        example for dropout in the non-model-parallel regions.
#         model-parallel state: This state is different among a set of model
#                               parallel GPUs, but the same across data parallel
#                               groups. This is used for example for dropout in
#                               model parallel regions.
#     """
#     # 2718 is just for fun and any POSITIVE value will work.
#     offset = seed + 2718
#     model_parallel_seed = offset + get_model_parallel_rank()
#     # Data parallel gets the original sedd.
#     data_parallel_seed = seed

#     if torch.distributed.get_rank() == 0:
#         print('> initializing model parallel cuda seeds on global rank {}, '
#               'model parallel rank {}, and data parallel rank {} with '
#               'model parallel seed: {} and data parallel seed: {}'.format(
#                   torch.distributed.get_rank(), get_model_parallel_rank(),
#                   get_data_parallel_rank(), model_parallel_seed,
#                   data_parallel_seed), flush=True)
#     _CUDA_RNG_STATE_TRACKER.reset()
#     # Set the default state.
#     torch.cuda.manual_seed(data_parallel_seed)
#     # and model parallel state.
#     _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                # model_parallel_seed)

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # mpu.model_parallel_cuda_manual_seed(seed)




set_random_seed(1111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

auto_loader = AutoLoader("lm",
                         model_name="ALM-1.0",
                         model_dir="./checkpoints")



model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()


# # tokenizer.EncodeAsIds('مقالة - سلعة: يُعد الصدق هو الحل الأفضل')
# [18375, 49837, 0, 21795, 49880, 22, 0, 89, 18187, 222, 1464, 3495]
model.to(device)
model.eval()
predictor = Predictor(model, tokenizer)

test_data = ["شرم الشيخ وجهة سياحية شهيرة [gMASK]"]
for text in test_data:
    print('===============================================\n')
    print(text, ":")
    for i in range(1):  #generate several times
        print("--------------sample %d :-------------------" % (i))
        print('-----------random sample: --------------')
        print(
            predictor.predict_generate_randomsample(text,
                                                    out_max_length=66,
                                                    top_k=10,
                                                    top_p=.1,
                                                    repetition_penalty=4.0,
                                                    temperature=1.2))
        print('-----------beam search: --------------')
        print(
            predictor.predict_generate_beamsearch(text,
                                                  out_max_length=66,
                                                  beam_size=10))
        print()
