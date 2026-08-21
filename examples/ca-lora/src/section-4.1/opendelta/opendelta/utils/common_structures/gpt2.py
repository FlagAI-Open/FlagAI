
Mappings = {}

Mappings['GPT2Model'] = {
    "wte": {"__name__":"embeddings"},
    "wpe": {"__name__":""},
    "h": {"__name__":"decoder.block",
        "$": {"__name__":"$",
            "attn": {"__name__":"attn",
                "c_attn": {"__name__":"q,k,v"},
                "c_proj": {"__name__":"proj"},
            },
            "ln_1": {"__name__":"attn.layer_norm"},
            "mlp":{ "__name__": "ff",
               "c_fc": {"__name__":"w1"},
               "c_proj": {"__name__":"w2"}
            },
            "ln_2": {"__name__":"ff.layer_norm"},
        },
    },
    "ln_f": {"__name__":"decoder.layer_norm"},
}