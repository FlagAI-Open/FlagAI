# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note "./checkpoints_poetry/{}/mp_rank_00_model_states.pt", {} is a directory in the checkpoints_poetry.
model_save_path = "/sharefs/baai-mrnd/xw/fork/FlagAI/examples/alm_seq2seq/checkpoints/ALM-1.0/pytorch_model.bin"

auto_loader = AutoLoader("seq2seq",
                         model_name="ALM-1.0",
                         model_dir="/sharefs/baai-mrnd/xw/fork/FlagAI/examples/alm_seq2seq/checkpoints/")
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

# model.load_state_dict(
#     torch.load(model_save_path, map_location=device)["module"])

model.to(device)
model.eval()
predictor = Predictor(model, tokenizer)

test_data = ['"مقالة - سلعة: يُعد الصدق هو الحل الأفضل']
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
