from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_info = "BAAI/AquilaChat2-70B"
tokenizer = AutoTokenizer.from_pretrained(model_info, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_info, trust_remote_code=True, torch_dtype=torch.bfloat16)
model.eval()
text = "请给出10个要到北京旅游的理由。"
from predict import predict
out = predict(model, text, tokenizer=tokenizer, max_gen_len=200, top_p=0.95,
              seed=1234, topk=100, temperature=0.9, sft=True,
              model_name="AquilaChat2-70B")
print(out)
