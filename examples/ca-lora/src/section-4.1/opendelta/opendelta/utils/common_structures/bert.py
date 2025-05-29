Mappings = {}

Mappings['BertModel'] = {
    "embeddings.word_embeddings": {"__name__":"embeddings"},
    "embeddings.position_embeddings": {"__name__":""},
    "embeddings.token_type_embeddings": {"__name__":""},
    "embeddings.LayerNorm": {"__name__":""},
    "encoder": {"__name__":"encoder",
        "layer": {"__name__":"block",
            "$": {"__name__":"$",
                "attention": {"__name__":"attn",
                    "self.query": {"__name__":"q"},
                    "self.key": {"__name__":"k"},
                    "self.value": {"__name__":"v"},
                    "output.dense": {"__name__":"proj"},
                    "output.LayerNorm": {"__name__":"layer_norm"},
                },
                "output": {"__name__":"ff",
                            "dense": {"__name__":"w2"},
                            "LayerNorm": {"__name__":"layer_norm"}
                },
                "intermediate": {"__name__":"ff",
                                "dense": {"__name__":"w1"},
                }
            }
        }
    },
}
