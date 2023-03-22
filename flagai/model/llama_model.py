import torch
from torch import nn
import os
from flagai.model.layers.feedforward import ColumnParallelLinear
from flagai.model.layers.embeddings import ParallelEmbedding
from flagai.model.blocks.llama_block import LLAMABlock, RMSNorm
from flagai.model.layers.attentions import precompute_freqs_cis
if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
    from flagai.mpu.random import checkpoint
elif os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint
import os 
from flagai.model.base_model import BaseModel 

class LLAMAConfig():
    r"""
    This is the configuration class to store the configuration of a [`~LLaMAModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LLaMAModel`] or [`~TFLLaMAModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from transformers import LLaMAModel, LLaMAConfig
    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LLaMAConfig()
    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LLaMAModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        dim=4096,
        max_seq_len=2048,
        max_batch_size=32,
        multiple_of=None,
        # intermediate_size=11008,
        n_layers=32,
        n_heads=32,
        #hidden_act="silu",
        #initializer_range=0.02,
 
        norm_eps=1e-6,
        use_cache=False,
        # pad_token_id=-1,
        # bos_token_id=0,
        # eos_token_id=1,
        # tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_batch_size = max_batch_size
        self.multiple_of = multiple_of
        
        # self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        # self.hidden_act = hidden_act
        # self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        # super().__init__(
        #     pad_token_id=pad_token_id,
        #     bos_token_id=bos_token_id,
        #     eos_token_id=eos_token_id,
        #     tie_word_embeddings=tie_word_embeddings,
        #     **kwargs,
        # )
def create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward      
        
class LLAMAModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.use_cache = config.use_cache
        self.config =config 
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.tok_embeddings = ParallelEmbedding(
            config.vocab_size,
            config.dim,
            init_method=lambda x: x)
       
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LLAMABlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        if os.getenv("ENV_TYPE") == "deepspeed+mpu":
            self.output = ColumnParallelLinear(
            config.dim, config.vocab_size, bias=False, init_method=lambda x: x
            )
        else:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, start_pos=0, labels=None, **kwargs):
        _bsz, seqlen = input_ids.shape
        if self.config.checkpoint_activations:
            h = checkpoint(create_custom_forward(self.tok_embeddings),input_ids)
        else:
            h = self.tok_embeddings(input_ids)
            
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        if self.config.checkpoint_activations:

            for layer in self.layers:
                h = checkpoint(create_custom_forward(layer),
                                h, start_pos, freqs_cis, mask, self.use_cache)
        else:
             for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask, self.use_cache)
      
        h = self.norm(h)
        if labels is not None:
            output = self.output(h)
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss_func(
                shift_logits.view(-1, self.config.vocab_size).contiguous(), shift_labels.view(-1).contiguous().long()).mean()
            
            return {
                'logits': self.output(h[:, -1, :]).float(), 
                'loss': loss,
                'hidden_states': output,
            }
        else :

            output = self.output(h[:, -1, :])  # only compute last logits
            return {
                "logits": output.float()
            }

    def load_weights(self, checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu")
        if "module" in sd:
            sd = sd["module"]
        self.load_state_dict(sd, strict=False)
        print(f"model config are loaded successfully...")
