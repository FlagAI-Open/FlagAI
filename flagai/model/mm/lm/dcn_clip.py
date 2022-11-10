import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import transformers
from models.lm.lgs.modeling_chclip_new import ChineseCLIP
from models.lm.lgs.modeling_chclip_new import CHCLIPProcess
import json

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class DCNCLIP3M768(AbstractEncoder):
    def __init__(self, version='ViT-b/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.ch_clip_model = ChineseCLIP.from_pretrained("/sharefs/baai-mrnd/zch/bs-bs-3m")
        self.ch_clip_model = self.ch_clip_model.eval()
        for param in self.ch_clip_model.parameters():
            param.requires_grad = False
        
        self.processor = CHCLIPProcess.from_pretrained("/sharefs/baai-mrnd/zch/bs-bs-3m")
        self.tokenizer = self.processor.tokenizer
        self.text_encoder = self.ch_clip_model.text_model
        self.layer_norm = LayerNorm(768)

    def forward(self, text):
        tokens = self.tokenizer(text,return_tensors='pt',padding=True).to(self.device)
        #tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=False,
        #                                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)
        z = self.text_encoder(**tokens)['hidden_states'][-1]
        z = self.layer_norm(z)
        
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        return z
    

class DCNCLIP30M1024(AbstractEncoder):

    def __init__(self, version='ViT-L/14', device="cuda", max_length=50, n_repeat=1, normalize=True):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.ch_clip_model = ChineseCLIP.from_pretrained("/sharefs/baai-mrnd/czz/lg30M")
        self.ch_clip_model = self.ch_clip_model.eval()

        for param in self.ch_clip_model.parameters():
            param.requires_grad = False
        
        self.processor = CHCLIPProcess.from_pretrained("/sharefs/baai-mrnd/czz/lg30M")
        self.tokenizer = self.processor.tokenizer

        self.text_encoder = self.ch_clip_model.text_model
        # self.normalize = normalize
        # self.layer_norm1 = LayerNorm(768)
        # self.layer_norm2 = LayerNorm(768)

        # self.proj = nn.Linear(1024,768)
        


    def forward(self, text):
        tokens = self.tokenizer(text,return_tensors='pt',truncation=True, 
        max_length=self.max_length, padding=True).to(self.device)
        #tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=False,
        #                                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)

        z = self.text_encoder(**tokens)['mimic_states']
        #  进行维度映射
        # z = self.layer_norm1(z)

        # z = self.proj(z)
        # z = self.layer_norm2(z)

        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        return z

class DCNCLIP3M1024(AbstractEncoder):

    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.ch_clip_model = ChineseCLIP.from_pretrained("/sharefs/baai-mrnd/zch/my_stable_diffusion/lm/lgs/lg-lg-8m")
        self.ch_clip_model = self.ch_clip_model.eval()

        for param in self.ch_clip_model.parameters():
            param.requires_grad = False
        
        self.processor = CHCLIPProcess.from_pretrained("/sharefs/baai-mrnd/zch/my_stable_diffusion/lm/lgs/lg-lg-8m")
        self.tokenizer = self.processor.tokenizer

        self.text_encoder = self.ch_clip_model.text_model
        self.normalize = normalize
        self.layer_norm1 = LayerNorm(1024)
        self.layer_norm2 = LayerNorm(768)

        self.proj = nn.Linear(1024,768)
        


    def forward(self, text):
        tokens = self.tokenizer(text,return_tensors='pt',padding=True).to(self.device)
        #tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=False,
        #                                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)

        z = self.text_encoder(**tokens)['hidden_states'][-1]
        #  进行维度映射
        z = self.layer_norm1(z)
        

        z = self.proj(z)
        z = self.layer_norm2(z)

        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        return z
