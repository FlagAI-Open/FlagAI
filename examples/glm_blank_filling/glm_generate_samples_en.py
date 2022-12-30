# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
import sys 
sys.path.append("/home/yanzhaodong/anhforth/FlagAI")
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import Tokenizer
from flagai.data.tokenizer.glm_large_en.glm_large_en_tokenizer import GLMLargeEnWordPieceTokenizer
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples')
    # Random seeds for reproducibility.
    # Model,
    # model_name = 'GLM-large-en'
    # model = GLMModel.from_pretrain(model_name=model_name,
    #                                download_path="./checkpoints/")
    # tokenizer = Tokenizer.from_pretrained(model_name)


    loader = AutoLoader(task_name='lm',
                                model_name='GLM-large-en',
                                only_download_config=False)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    # tokenizer = GLMLargeEnWordPieceTokenizer()
    # import pdb;pdb.set_trace()
    model.cuda(torch.cuda.current_device())

    # predictor = Predictor(model, tokenizer)

    predictor = Predictor(model, tokenizer)
    # generate samples
    text = [
        'Question: Is drinking beer bad for your health? Answer: [gMASK]',
    ]
    # text = [
    #     'Question: Is fruit good for your health? Answer: [gMASK]',
    # ]
    for t in text:
        output = predictor.predict_generate_randomsample(
            t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        # output = predictor.predict_generate_beamsearch(t,
        #                                         out_max_length=66,
        #                                         beam_size=10)
        print(t, '\n', output)

 
