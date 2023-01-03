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

    loader = AutoLoader(task_name='lm',
                            model_name='GLM-large-en-generation',
                            only_download_config=False)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    model.cuda(torch.cuda.current_device())
    predictor = Predictor(model, tokenizer)
    text = [
        'Is drinking beer bad for your health?',
    ]
    for t in text:
        output = predictor.predict_generate_randomsample(
            t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t, '\n', output)

 
