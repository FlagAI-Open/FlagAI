(commonstructure)=

# Common Structure Mapping

```{figure} ../imgs/transformers_structure.png
:width: 400px
:name: transformers_structure
```

Although different PTMs often share similar Transformers structures, the codebases, and most importantly, the variable names for each submodule, are quite different.



On the one hand, we **encourage the users to first [visualize](visualization) the PTMs' structure and then determine the name of submoduels.**

On the other hand, we designed a unified name convention of Transformer Structure, and provided several structure mapping from the original name to the unified name convention. 

In this section, we will illustrate the unified name convention and structure mapping.


## Common blocks in Transformers structure.


- embeddings (word embedding)
- encoder
  - block
    - $ (layer_id)
      - attn
        - q, k, v
        - proj
        - layer_norm
      - ff
        - w1
        - w2
        - layer_norm
- decoder (similar to encoder)
- lm_head
  - proj

Visualize bert-base using a common structure name: The submodules that are not common are grey.

```{figure} ../imgs/commonstructure_vis.png
:width: 600px
:name: commonstructure_vis
```

(mappingexample)=
## Example

Example of bert mapping: a tree with node names specified by <span style="font-weight:bold;color:rgb(55, 125, 34);" >"\_\_name\_\_"</span>
```json
{
    "bert.embeddings.word_embeddings": {"__name__":"embeddings"},
    "bert.embeddings.position_embeddings": {"__name__":""},
    "bert.embeddings.token_type_embeddings": {"__name__":""},
    "bert.embeddings.LayerNorm": {"__name__":""},
    "bert.encoder": {"__name__":"encoder",
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
                "intermediate.dense": {"__name__":"ff.w1"},
            }
        }
    },
    "cls.predictions": {"__name__": "lm_head",
        "transform.dense": {"__name__":""},
        "transform.LayerNorm": {"__name__":""},
        "decoder": {"__name__":"proj"},
    }
}
```

