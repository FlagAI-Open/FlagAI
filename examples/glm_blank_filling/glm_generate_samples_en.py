# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import torch
from flagai.model.predictor.predictor import Predictor
from flagai.auto_model.auto_loader import AutoLoader
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples')
    # Random seeds for reproducibility.
    # Model,
    loader = AutoLoader(task_name='lm',
                                model_name='GLM-large-en',
                                only_download_config=False)
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    model.cuda(torch.cuda.current_device())

    predictor = Predictor(model, tokenizer)
    # generate samples
    text = [
        'Question: Is drinking beer bad for your health? Answer: [gMASK]',
    ]
    for t in text:
        output = predictor.predict_generate_randomsample(
            t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t, '\n', output)

 
