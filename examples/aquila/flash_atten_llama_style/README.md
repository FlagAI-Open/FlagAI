# Numerical differences when converting flash\_attn.models.gpt into flagai.model.llama\_model.LLAMAModel
Three modules are the main influencing factors as follows:
- flash\_attn.ops.rms\_norm should be switched on.
- RotaryEmbedding should be float32.
- QKV Linear layer should be integrated into one.

# How to convert
- Enable flash\_atten\_llama\_style in LLAMAModel config.
- Make sure that RotaryEmbedding should be float32 specially after type casting such half or float16.
- When load model, QKV Linear weights should be concated.

# Test Case
python test\_infe.py
```
import torch
logits_flagai = torch.load('logits_flagai.pt')
logits_flash = torch.load('logits_flash.pt')
diff = logits_flagai != logits_flash
torch.any(diff)
```
