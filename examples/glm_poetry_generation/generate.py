# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import sys
sys.path.append("/mnt/wchh/FlagAI-internal")
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note "./checkpoints_poetry/{}/mp_rank_00_model_states.pt", {} is a directory in the checkpoints_poetry.
model_save_path = "/mnt/finetune_models/glm_poetry/mp_rank_00_model_states.pt"#"./checkpoints_poetry/1/mp_rank_00_model_states.pt"

auto_loader = AutoLoader("seq2seq",
                         model_name="GLM-large-ch",
                         model_dir="/data/chkpt/")
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()

model.load_state_dict(
    torch.load(model_save_path, map_location=device)["module"])

model.to(device)
model.eval()
predictor = Predictor(model, tokenizer)

test_data = ['初夏：五言绝句', '桃花：七言绝句', '秋思：五言律诗', '边塞：七言律诗']
for text in test_data:
    print('===============================================\n')
    print(text, ":")
    for i in range(4):  #generate several times
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
