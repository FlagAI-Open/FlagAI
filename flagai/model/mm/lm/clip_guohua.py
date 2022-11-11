import torch
import torch.nn as nn
from flagai.model.mm.clip_guohua.model import CLIP
from flagai.model.base_model import BaseModel


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


class CN_CLIP(BaseModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        max_length = config["max_length"]
        normalize = config["normalize"]
        self.tokenizer = kwargs.get("tokenizer")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.model = CLIP(config, **kwargs)

        # 冻结权重
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.normalize = normalize
        self.layer_norm1 = LayerNorm(768)
        self.layer_norm2 = LayerNorm(768)

        self.proj = nn.Linear(768, 768)

    def forward(self, text):
        text = self.tokenizer.tokenize(text).to(self.device)
        z = self.model.encode_text(text)
        z = self.layer_norm1(z)
        z = self.proj(z)
        z = self.layer_norm2(z)

        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        return z

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        self.model.load_state_dict(sd)
