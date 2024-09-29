# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import openai
import json
import time
import random
import os
import requests
import base64

prompt_template = '''Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to college levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students. 
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or college. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
<{EXAMPLE}>.

After examining the extract: 
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score:  <total points>"'''

def get_from_gpt4(ques, engine="gpt-4"):
  uer_prompt = prompt_template.format(EXAMPLE=ques)
  messages = []
  messages.append({"role": "user", "content": uer_prompt})

  openai_api_key = "EMPTY"
  openai_api_base = "http://localhost:8000/v1"
  openai_api_base = "http://120.92.91.62:9301/v1"

  from openai import OpenAI
  client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
  )

  models = client.models.list()
  model = models.data[0].id

  rsp = None
  try:
    rsp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
  except:
    import traceback
    traceback.print_exc()
    pass
  #return rsp["choices"][0]["message"]["content"]
  return rsp

def parse_score_from_review(review):
    try:
        import re
        match = re.search(r'Educational score:\s+(\d+\.?\d*)', review)
        if match:
          return float(match.group(1))
        else:
          print(f"Failed to parse scores from {review}")
          return -1
    except:
        print(f"Failed to parse scores from {review}")
        return -1

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--file-path', type=str, default="", help="file path", required=True)
  args = parser.parse_args()

  import jsonlines
  file_path = args.file_path
  output_file_path = f'{file_path}.jsonl.qwen'
  writer = jsonlines.open(output_file_path, mode='w')
  with jsonlines.open(file_path) as reader:
    for line in reader:
      ques = line['text']
      score = -1.0
      response = get_from_gpt4(ques, engine='Qwen2-72B-Instruct')
      content = ''
      print(response)
      if response is None:
        pass
      else:
        for choice in response.choices:
            content = choice.message.content
            score = parse_score_from_review(content)
            break
      line['score'] = score
      line['content'] = content
      writer.write(line)

