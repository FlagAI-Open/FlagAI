from flagai.model.layers.attentions import GPT2Attention
from flagai.model.layers.feedforward import GPT2MLP
from torch import nn


class GPT2Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config['n_embd']
        inner_dim = config['n_inner'] if config[
            'n_inner'] is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config['layer_norm_epsilon'])
        self.attn = GPT2Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config['layer_norm_epsilon'])
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
