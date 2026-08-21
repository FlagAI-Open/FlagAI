import torch
import bmtrain as bmt
from layers import TransformerEncoder, Layernorm, Embedding, TransformerEncoder
from bmtrain.global_var import config

class GPT(bmt.DistributedModule):
    def __init__(self,
            num_layers : int, vocab_size : int,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            max_distance : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.max_distance = max_distance

        if config["tp_size"] > 1:
            self.word_emb = bmt.nn.VPEmbedding(vocab_size, dim_model, dtype=dtype)
        else:
            self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
        self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        
        if config['pipe_size'] > 1:
            self.transformers = bmt.PipelineTransformerBlockList([
                bmt.Block(
                    TransformerEncoder(
                        dim_model, dim_head, num_heads, dim_ff, bias, dtype
                    )
                    , mode="PIPE"
                )
                for _ in range(num_layers)
            ])
        else:
            self.transformers = bmt.TransformerBlockList([
                bmt.Block(
                    TransformerEncoder(
                        dim_model, dim_head, num_heads, dim_ff, bias, dtype
                    )
                )
                for _ in range(num_layers)
            ])

        self.layernorm = Layernorm(dim_model, dtype=dtype)

    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
        ) -> torch.Tensor:

        mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])
        if config["tp_size"] > 1:
            input = input.chunk(config["tp_size"], dim=1)[config["tp_rank"]]
            pos = pos.chunk(config["tp_size"], dim=1)[config["tp_rank"]]    
        out = self.pos_emb(pos) + self.word_emb(input)

        # for layer in self.transformers:
        out = self.transformers(out, mask_2d, None)
        out = self.layernorm(out)
        logits = self.word_emb(out, projection=True)
        bmt.inspect.record_tensor(logits, "logits")

        return logits
