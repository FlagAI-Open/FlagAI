#!pip install transformers>=4.30.2
#!pip install optimum>=1.8.7 optimum-intel[openvino]>=1.9.0
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('sammysun0711/aquilachat-7b-hf')
model = AutoModelForCausalLM.from_pretrained('sammysun0711/aquilachat-7b-hf', trust_remote_code=True)
model = model.eval()
# from optimum.onnxruntime import ORTModelForCausalLM
# model = ORTModelForCausalLM.from_pretrained('sammysun0711/aquilachat-7b-hf', export=True, use_cache=True, trust_remote_code=True)

# from optimum.intel import OVModelForCausalLM
# model = OVModelForCausalLM.from_pretrained('sammysun0711/aquilachat-7b-hf', export=True, use_cache=True, trust_remote_code=True)

question = '北京为什么是中国的首都？'
prompt = (
    '''A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.'''
    f'''###Human: {question}###Assistant:'''
)
with torch.no_grad():
    ret = model.generate(
        **tokenizer(prompt, return_tensors='pt').to('cpu'),
        do_sample=False,
        max_new_tokens=200,
        use_cache=True
    )
    print(tokenizer.decode(output_ids))