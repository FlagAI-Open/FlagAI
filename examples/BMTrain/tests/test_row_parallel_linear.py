import torch
import bmtrain as bmt
from bmtrain.global_var import config
import numpy as np

def run_bmt(x, ckp_path, split_input=True, use_checkpoint_block=True):
    linear = bmt.nn.RowParallelLinear(8,8, split_input=split_input, all_reduce_output=True)
    if use_checkpoint_block:
        linear = bmt.Block(linear)
    bmt.init_parameters(linear)
    y = linear(x)
    y.sum().backward()
    bmt.save(linear, ckp_path)
    bmt.synchronize()
    return y, linear._parameters['weight'].grad, linear._parameters['bias'].grad

def run_torch(x, ckp_path):
    linear = torch.nn.Linear(8, 8)
    linear_dict = torch.load(ckp_path)
    linear.load_state_dict(linear_dict)
    linear = linear.cuda()
    linear.weight.requires_grad_()
    y = linear(x)
    y.sum().backward()
    return y, linear.weight.grad, linear.bias.grad

def run(split_input, use_checkpoint_block, ckp_path):
    tp_size = bmt.config['tp_size']
    torch.cuda.manual_seed(100)
    tp_rank = config['topology'].tp_id
    x = torch.randn(8,8, device='cuda').requires_grad_()
    rank_x = x.chunk(tp_size, dim=0 if split_input else 1)[tp_rank]
    y1, weight_grad1, bias_grad1 = run_bmt(rank_x, ckp_path, split_input, use_checkpoint_block)
    y2, weight_grad2, bias_grad2 = run_torch(x, ckp_path)
    np.testing.assert_allclose(y1.detach().cpu().numpy(), y2.detach().cpu().numpy(), atol=1e-5)

    weight_grad_list = weight_grad2.chunk(tp_size, dim=1)
    assert np.allclose(weight_grad1.reshape(weight_grad_list[tp_rank].shape).cpu().numpy(), weight_grad_list[tp_rank].cpu().numpy())

    assert np.allclose(bias_grad1.cpu().numpy(), bias_grad2.cpu().numpy())

def test_split_input():
    run(True, False, 'row_parallel_linear.ckp')
    run(True, True, 'row_parallel_linear.ckp')

def test_no_split_input():
    run(False, False, 'row_parallel_linear_no_split.ckp')
    run(False, True, 'row_parallel_linear_no_split.ckp')

if __name__ == "__main__":
    bmt.init_distributed(tp_size=2)
    test_no_split_input()
    test_split_input()

