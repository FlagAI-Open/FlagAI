# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
import torch

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

if __name__ == "__main__":
    loader = AutoLoader(task_name="lm",
                        model_name="GLM-large-ch",
                        )
    model = loader.get_model().half()
    tokenizer = loader.get_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictor = Predictor(model, tokenizer)

    text = ['问题：啤酒伤胃吗？回答：[gMASK]',
            "问题：隔夜菜能吃吗？回答：[gMASK]",
            "问题：如何评价许嵩？回答：[gMASK]"]
    for t in text:
        output=predictor.predict_generate_randomsample(t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t,'\n',output)

    text = ['北京故宫是中国[MASK]非物质文化遗产。',
            "上海是中国[MASK]大都市。",
            "天津大学是[MASK]现代大学。"]
    for t in text:
        output = predictor.predict_generate_randomsample(t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t, '\n', output)
    #
    text = ["人工智能是一个以计算机科学为基础，由计算机、数学、哲学等多学科交叉融合的交叉学科，[sMASK]，具有非常巨大的前景。",
            "最近十多年来，人工神经网络的研究工作不断深入，已经取得了很大的进展，[sMASK]，表现出了良好的智能特性。"]
    for t in text:
        output = predictor.predict_generate_randomsample(t, top_k=50, repetition_penalty=4.0, top_p=1.0)
        print(t, '\n', output)
