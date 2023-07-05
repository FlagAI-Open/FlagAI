import torch
from torch import nn
import os
from flagai.model.layers.feedforward import ColumnParallelLinear
from flagai.model.layers.embeddings import ParallelEmbedding
from flagai.model.blocks.aquila_block import AQUILABlock, RMSNorm
from flagai.model.layers.attentions import precompute_freqs_cis
from flagai.model.utils import normal_init_method
if os.getenv('ENV_TYPE') == 'deepspeed+mpu':
    from flagai.mpu.random import checkpoint
elif os.getenv('ENV_TYPE') == 'deepspeed':
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
else:
    from torch.utils.checkpoint import checkpoint
import os 
from flagai.model.base_model import BaseModel

class AQUILAConfig(dict):

    model_type = "aquila"

    def __init__(
        self,
        vocab_size=32000,
        dim=4096,
        max_seq_len=2048,
        max_batch_size=1,
        multiple_of=None,
        # intermediate_size=11008,
        n_layers=32,
        n_heads=32,
        #hidden_act="silu",
        initializer_range=0.02,
        checkpoint_activations=False,
 
        norm_eps=1e-6,
        use_cache=False,
        flash_atten=False,
        flash_atten_pdrop=0.0,
        ignore_index=-100,
        bmt_comm_overlap=False,
        bmt_fused_ce=False,
        bmt_fused_ce_inplace=True,
        fix_large_bsz=False,
        # pad_token_id=-1,
        # bos_token_id=0,
        # eos_token_id=1,
        # tie_word_embeddings=False,
        flash_atten_aquila_style=False,
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

        self.initializer_range = initializer_range
        self.checkpoint_activations = checkpoint_activations

        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.flash_atten = flash_atten
        self.flash_atten_pdrop = flash_atten_pdrop
        self.ignore_index = ignore_index
        self.bmt_comm_overlap = bmt_comm_overlap
        self.bmt_fused_ce = bmt_fused_ce
        self.bmt_fused_ce_inplace = bmt_fused_ce_inplace
        self.fix_large_bsz = fix_large_bsz

        self.flash_atten_aquila_style = flash_atten_aquila_style

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
        
class AQUILAModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = AQUILAConfig()
        for key in config.json_config:
            if hasattr(self.config, key):
                setattr(self.config, key, config.json_config[key])
        config = self.config
        
        self.use_cache = config.use_cache
        print("***************use cache", self.use_cache)
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        if os.getenv("ENV_TYPE") == 'deepspeed+mpu':
            self.tok_embeddings = ParallelEmbedding(
                config.vocab_size,
                config.dim,
                init_method=normal_init_method(0, self.config.initializer_range))
        else:
            self.tok_embeddings = nn.Embedding(
                config.vocab_size,
                config.dim,
            )
            init_method=normal_init_method(0, self.config.initializer_range)
            init_method(self.tok_embeddings.weight)

        self.start_pos = 0
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(AQUILABlock(layer_id, config))

        if config.flash_atten_aquila_style:
            import flash_attn
            self.norm = flash_attn.ops.rms_norm.RMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        if os.getenv("ENV_TYPE") == "deepspeed+mpu":
            self.output = ColumnParallelLinear(
                config.dim, config.vocab_size, bias=False,
                init_method=normal_init_method(0, self.config.initializer_range)
            )
        else:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
            init_method=normal_init_method(0, self.config.initializer_range)
            init_method(self.output.weight)

        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2
        )

        if os.getenv("ENV_TYPE") == "bmtrain" and self.config.bmt_fused_ce:
            import bmtrain as bmt
            self.loss_func = bmt.loss.FusedCrossEntropy(
                ignore_index=self.config.ignore_index,
                inplace=self.config.bmt_fused_ce_inplace)
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)


    def pre_train_hook(self):
        """ before training """
        if os.getenv("ENV_TYPE") == "bmtrain" and self.config.bmt_comm_overlap:
            import bmtrain as bmt
            blocks = [layer for layer in self.layers]
            self.layers = bmt.TransformerBlockList(blocks)

    def forward(self, input_ids: torch.Tensor, start_pos=0, labels=None, **kwargs):
        
        _bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
            
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        self.start_pos = start_pos
        if self.config.checkpoint_activations:
            for layer in self.layers:
                layer.use_cache = self.use_cache
                layer.start_pos = start_pos
                h = checkpoint(create_custom_forward(layer), h, freqs_cis, mask)
        elif os.getenv("ENV_TYPE") == "bmtrain" and self.config.bmt_comm_overlap:
            # to overlap communication with computation
            for layer in self.layers:
                layer.use_cache = self.use_cache
                layer.start_pos = start_pos
                
            h = self.layers(h, freqs_cis, mask)
        else:
            for layer in self.layers:
                layer.use_cache = self.use_cache
                layer.start_pos = start_pos
                h = layer(h, freqs_cis, mask)
                
        # import pdb;pdb.set_trace()
        h = self.norm(h)
        if labels is not None:
            h = self.output(h)

            shift_logits = h[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            bsz_half = _bsz // 2
            bsz_split = bsz_half * seqlen
            bsz_total = _bsz * seqlen
            ## torch 1.12.1 
            ## CUDA Illegal memory access on CrossEntropyLoss with large batch size
            ## https://github.com/pytorch/pytorch/issues/85005
            if self.config.fix_large_bsz and bsz_total > bsz_split:
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1).long()
                loss_split = self.loss_func(shift_logits[:bsz_split, :], shift_labels[:bsz_split]).mean()
                loss_remain = self.loss_func(shift_logits[bsz_split:, :], shift_labels[bsz_split:]).mean()
                bsz_remain = bsz_total - bsz_split
                ## NaN
                #loss = (loss_split * bsz_split + loss_remain * bsz_remain) / bsz_total
                loss = bsz_split / bsz_total * loss_split + bsz_remain / bsz_total * loss_remain
            else:
                loss = self.loss_func(
                    shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1).long()).mean()
            
            return {
                'logits': h, 
                'loss': loss,
                'hidden_states': h,
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
        print(f"model checkpoint_path={checkpoint_path} are loaded successfully...")
