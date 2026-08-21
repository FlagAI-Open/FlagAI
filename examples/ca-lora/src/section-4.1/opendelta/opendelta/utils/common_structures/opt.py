

Mappings = {}
Mappings['OPTModel'] = {
    "decoder.embed_tokens": {"__name__":"embeddings"},
    "decoder.embed_positions": {"__name__":""},
    "decoder.project_out": {"__name__":""},
    "decoder.project_in": {"__name__":""},
    "decoder": {"__name__":"decoder",
        "layers": {"__name__":"block",
            "$": {"__name__":"$",
                "self_attn": {"__name__":"attn",
                    "q_proj": {"__name__":"q"},
                    "k_proj": {"__name__":"k"},
                    "v_proj": {"__name__":"v"},
                    "out_proj": {"__name__":"proj"}
                },
                "self_attn_layer_norm": {"__name__":"layer_norm"},
                "fc1": {"__name__":"ff.w1", "__virtual__": "ff", "__order__": "first"},
                "fc2": {"__name__":"ff.w2","__virtual__": "ff", "__order__": "last"},
                "final_layer_norm": {"__name__":"layer_norm"},
            }
        }
    }
}