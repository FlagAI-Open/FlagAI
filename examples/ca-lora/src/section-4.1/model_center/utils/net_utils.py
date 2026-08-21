# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
import os
import requests
import tqdm
import bmtrain as bmt

file_names = {
    'config': ['config.json'],
    'model': ['pytorch_model.pt'],
    'tokenizer': ['vocab.json', 'vocab.txt', 'merges.txt', 'tokenizer.json', 'added_tokens.json', 'special_tokens_map.json', 'tokenizer_config.json', 'spiece.model', 'vocab.model'],
}

def download(path, url):
    req = requests.get(url, stream=True)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file = open(path, "wb")
        req.raise_for_status()
        print(f"download from web, cache will be save to: {path}")
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm.tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=total,
            desc="Downloading",
        )
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                file.write(chunk)
        progress.close()
        file.close()
    except:
        file.close()
        os.remove(path)

def check_web_and_convert_path(path, load_type): # TODO add hash
    if os.path.isdir(path):
        try:
            bmt.print_rank(f"load from local file: {path}")
        except:
            pass
        return path
    else:
        if bmt.rank() == 0:
            url = f"https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/{path}"
            try:
                requests.get(f'{url}/config.json', stream=True).raise_for_status() # use config.json to check if identifier is valid
            except:
                raise ValueError(f"'{path}' is not a valid model identifier")
            cache_path = os.path.expanduser(f"~/.cache/model_center/{path}")
            for name in file_names[load_type]:
                p = os.path.join(cache_path, name)
                if os.path.exists(p):
                    bmt.print_rank(f"load from cache: {p}")
                else:
                    if bmt.rank() == 0:
                        download(p, f"{url}/{name}")
        else:
            cache_path = os.path.expanduser(f"~/.cache/model_center/{path}")
        try:
            bmt.synchronize()
        except:
            pass
        return cache_path
        
