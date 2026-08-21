import os
import torch
import numpy as np
import types
import bmtrain as bmt

class BMMoE:
    '''
    BMMoE replaces the feed-forward modules in PLMs with MoE simulation modules.
    '''

    @staticmethod
    def get_hidden(model, config, forward_fn):
        '''
        Get the hidden states of the model.

        `foward_fn` should have the following arguments: `foward_fn(model, enc_input, enc_length, dec_input, dec_length, targets, loss_func)`. These arguments are general for existing Transformers. For decoder-only model, `enc_input` and `enc_length` can be set to None. For encoder-only model, `dec_input` and `dec_length` can be set to None. Similarly, `student` and `teacher` models also have the following arguments: `model(enc_input, enc_length, dec_input, dec_length)`.

        :param model: Model to get the hidden states.
        :param config: Configuration of getting the hidden states. It should contain the names of the layernorm modules before MoEfied FFNs.
        :param forward_fn: Forward function. 
        '''
        moe_config = config.get('MoEfication')
        if not moe_config['is_moefy']:
            return forward_fn

        modules = get_modified_modules(model, moe_config['first_FFN_module'])

        update_forward(modules)

        def forward(model, loss_func, targets, *model_args, **model_kwargs):
            with bmt.inspect.inspect_tensor() as inspector:
                outputs = forward_fn(
                    model, loss_func, targets, *model_args, **model_kwargs)
            
            records = {}
            for record in inspector._summary:
                if 'moe_hidden' in record['name']:
                    records[record['name']] = record['tensor']
            
            outputs[5] = records
            return outputs
        return forward

    @staticmethod
    def moefy(model, num_expert, topk, num_layer, patterns, checkpoint):
        '''
        Replace the feed-forward modules in PLMs with MoE modules according to the results of MoEfication from the checkpoint file.

        :param model: Model to MoEfy.
        :param num_expert: Number of experts.
        :param topk: Top-k for each expert.
        :param checkpoint: Path to load the MoEfication results.
        '''
        # after parameter initialization

        for pattern in patterns:
            for layer_idx in range(num_layer):
                target = pattern.format(layer_idx)
                ffn = None
                ffn_name = '.'.join(target.split('.')[:-1])
                for k, v in model.named_modules():
                    if k == ffn_name:
                        ffn = v
                        break

                assert ffn is not None

                path = os.path.join(checkpoint, target+'_input_compl')

                assert os.path.exists(path)

                ffn.moe = True
                ffn.layer_idx = layer_idx

                ffn.mlp = torch.load(path).to("cuda:{}".format(torch.cuda.current_device()))

                label_file = os.path.join(checkpoint, target)
                labels = torch.load(label_file)
                cluster_num = max(labels)+1
                assert cluster_num == num_expert
                patterns = []
                for i in range(cluster_num):
                    patterns.append(np.array(labels) == i)
                patterns = np.array(patterns)
                ffn.patterns = torch.Tensor(patterns).cuda()

                ffn.k = topk

                def _forward(module_self, x):
                    bsz, seq_len, hidden_size = x.shape
                    hidden_states_mlp = x.clone().detach().float()
                    hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)
                    hidden_states_mlp = hidden_states_mlp / torch.norm(hidden_states_mlp, dim=-1).unsqueeze(-1)
                    score = module_self.mlp(hidden_states_mlp)
                    labels = torch.topk(score, k=module_self.k, dim=-1)[1].view(bsz, seq_len, module_self.k)
                    cur_mask = torch.nn.functional.embedding(labels, module_self.patterns).sum(-2)

                    x = module_self.forward_old(x)

                    x[cur_mask == False] = 0

                    return x

                ffn.forward_old = ffn.forward
                ffn.forward = types.MethodType(_forward, ffn)

def get_modified_modules(model, first_FFN_module):
    '''
    Get the modules that are modified by MoEfication.

    :param model: Model to get the modified modules.
    :param first_FFN_module: The index of the first feed-forward module.
    :return: The modules that are modified by MoEfication.
    '''
    modules = []
    for name, module in model.named_modules():
        if any([x in name for x in first_FFN_module]):
            modules.append(module)
    return modules

def update_forward(modules):
    inspect_name = "moe_hidden"
    def _forward(module_self, x):
        x = module_self.forward_old(x)
        bmt.inspect.record_tensor(x, inspect_name)
        return x
    
    for module in modules:
        module.forward_old = module.forward
        module.forward = types.MethodType(_forward, module)