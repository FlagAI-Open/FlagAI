Mappings = {}

t5encoder =  {"__name__":"encoder",
        "embed_tokens": {"__name__":"embeddings"},
        "block": {"__name__":"block",
            "$": {"__name__":"$",
                "layer.0": {"__name__":"attn",
                    "SelfAttention.q": {"__name__":"q"},
                    "SelfAttention.k": {"__name__":"k"},
                    "SelfAttention.v": {"__name__":"v"},
                    "SelfAttention.o": {"__name__":"proj"},
                    "SelfAttention.relative_attention_bias": {"__name__":""},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.1": {"__name__":"ff",
                    "DenseReluDense.wi": {"__name__":"w1"},
                    "layer_norm": {"__name__":"layer_norm"},
                    "DenseReluDense.wo": {"__name__":"w2"},
                }
            }
        },
        "final_layer_norm": {"__name__":"layer_norm"},
    }

t5decoder = {"__name__":"decoder",
        "embed_tokens": {"__name__":"embeddings"},
        "block": {"__name__":"block",
            "$": {"__name__":"$",
                "layer.0": {"__name__":"attn",
                    "SelfAttention.q": {"__name__":"q"},
                    "SelfAttention.k": {"__name__":"k"},
                    "SelfAttention.v": {"__name__":"v"},
                    "SelfAttention.o": {"__name__":"proj"},
                    "SelfAttention.relative_attention_bias": {"__name__":""},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.1": {"__name__":"crossattn",
                    "EncDecAttention.q": {"__name__":"q"},
                    "EncDecAttention.k": {"__name__":"k"},
                    "EncDecAttention.v": {"__name__":"v"},
                    "EncDecAttention.o": {"__name__":"proj"},
                    "layer_norm": {"__name__":"layer_norm"},
                },
                "layer.2": {"__name__":"ff",
                    "DenseReluDense.wi": {"__name__":"w1"},
                    "layer_norm": {"__name__":"layer_norm"},
                    "DenseReluDense.wo": {"__name__":"w2"},
                }
            }
        },
        "final_layer_norm": {"__name__":"layer_norm"},
    }



Mappings['T5Model'] =  {
    "shared": {"__name__":"embeddings"},
    "encoder": t5encoder,
    "decoder": t5decoder, 
}

Mappings['T5ForConditionalGeneration'] =  {
    "shared": {"__name__":"embeddings"},
    "encoder": t5encoder,
    "decoder": t5decoder, 
}

Mappings['T5EncoderModel'] =  {
    "shared": {"__name__":"embeddings"},
    "encoder": t5encoder,
}