# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""
import sys
sys.path.append('/data/wang/models/GLMgeneration')
sys.path.append('/data/wang/models/FlagAI')
import os
import torch

from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import GLMLargeChTokenizer
from flagai.model.predictor.predictor import Predictor
if __name__ == "__main__":
    """Main training program."""
    print('Generate Samples')
    # Random seeds for reproducability.

    tokenizer = GLMLargeChTokenizer(vocab_path='./state_dict/GLM-large-ch/cog-pretrain.model',
                                    add_block_symbols=True,
                                    add_task_mask=True,
                                    add_decoder_mask=False,
                                    fix_command_token=False)

    # Model,
    model = GLMModel.from_pretrain(model_name='GLM-large-ch', download_path="./state_dict/")
    model.cuda(torch.cuda.current_device())

    predictor = Predictor(model, tokenizer)
    # generate samples
    text = '问题：啤酒伤胃吗？回答：[gMASK]'
    output=predictor.predict_generate_randomsample(text)
    print(text,'\n',output)

    text = '北京故宫是中国[MASK]非物质文化遗产。'
    output = predictor.predict_generate_randomsample(text)
    print(text, '\n', output)
    #
    text = '人工智能是一个以计算机科学为基础，由计算机、数学、哲学等多学科交叉融合的交叉学科，[sMASK]，具有非常巨大的前景。'
    output = predictor.predict_generate_randomsample(text)
    print(text, '\n', output)
