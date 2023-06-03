# -*- coding: utf-8 -*-
import json
import requests

# prompt = "Question: Aristotle says  that what makes things be what they are--their essence--does not exist apart from individ-uals that exist in the world.  So if all the members of a species were destroyed, then their essence or form:\nA. would likewise be destroyed.\nB. would be destroyed only if there were no one around to remember the species.\nC. would continue existing (as with Plato's Forms) in some other realm of being.\nD. would not be destroyed because there was no essence or form originally to be destroyed; there are only individuals, not universal essences or natures of things.\nAnswer:"
#prompt = "如何评价许嵩？"
#prompt = "1+1="
#prompt = "你是谁做的？"
#prompt = "你是谁？请介绍一下你自己。"
#prompt = "简单介绍智源研究院。"
#prompt = "请给出10个要到北京旅游的理由。"
prompt = "请问1+1="

raw_request = {
            "engine": 'glm-130b',
            "prompt": prompt,
            "temperature": 0.9,
            "num_return_sequences": 1,
            "max_new_tokens": 128,
            "top_p": 0.95,
            "echo_prompt": False,
            "top_k_per_token": 200,
            "stop_sequences": [],
            "seed": 100,
        }

url = 'http://47.100.10.8:5063/func'
url = 'http://47.100.10.8:7870/func'
#url = 'http://47.100.10.8:5062/func'
data_json = json.dumps(raw_request)
# requests.post(url, json=data_json)
response = requests.post(url, json=data_json)
print(response)
result = response.json()
print(result)
