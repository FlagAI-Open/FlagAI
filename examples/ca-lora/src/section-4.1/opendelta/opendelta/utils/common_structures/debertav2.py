
Mappings = {}

Mappings['DebertaV2Model'] = {
    "embeddings.word_embeddings": {"__name__":"embeddings"},
    "embeddings.LayerNorm": {"__name__":""},
    "encoder": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "self.query_proj": {"__name__":"q"},
                    "self.key_proj": {"__name__":"k"},
                    "self.value_proj": {"__name__":"v"},
                    "output.dense": {"__name__":"proj"},
                    "output.LayerNorm": {"__name__":"layer_norm"},
                },
                "output": {"__name__":"ff",
                            "dense": {"__name__":"w2"},
                            "LayerNorm": {"__name__":"layer_norm"}
                },
                "intermediate.dense": {"__name__":"ff.w1"},
            }
        },
        "rel_embeddings": {"__name__": ""},
        "LayerNorm": {"__name__": ""},
        "conv": {"__name__": "",
            "conv": {"__name__": ""},
            "LayerNorm": {"__name__": ""}
        }
    },
}