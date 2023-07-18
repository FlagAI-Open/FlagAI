# -*- coding: utf-8 -*-
import json
import requests

prompt = "pytorch介绍"
# prompt = "介绍几个生僻字"
url = 'http://127.0.0.1:7860/func'
# url = 'http://127.0.0.1:7888/stream_func'

raw_request = {
        "prompt": prompt,
        "top_k_per_token": 100,
        "sft": True,
        }

data_json = json.dumps(raw_request)
response = requests.post(url, json=data_json)
print(response)
result = response.json()
print(result)
