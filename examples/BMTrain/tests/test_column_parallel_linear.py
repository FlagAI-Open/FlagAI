import torch
import bmtrain as bmt
from bmtrain.global_var import config
import numpy as np

def run_bmt(x, gather_input, gather_output, ckp_path, tp_size=2):
    linear = bmt.nn.ColumnParallelLinear(8,8, gather_input=gather_input, gather_output=gather_output)
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

def run(gather_input, gather_output, ckp_path):
    torch.cuda.manual_seed(100)
    tp_size = config["tp_size"]
    tp_rank = config['topology'].tp_id
    x = torch.randn(8, 8, 8, device='cuda')
    bmt_x = x.clone()
    if gather_input:
        rank_x = bmt_x.chunk(tp_size, dim=0)[tp_rank]
    else:
        rank_x = bmt_x
    rank_x.requires_grad_()
    x.requires_grad_()
    y1, weight_grad1, bias_grad1 = run_bmt(rank_x, gather_input, gather_output, ckp_path)
    y2, weight_grad2, bias_grad2 = run_torch(x, ckp_path)
    tp_rank = config['topology'].tp_id
    if gather_output:
        assert np.allclose(y1.detach().cpu().numpy(), y2.detach().cpu().numpy())
    else:
        torch_out_list = torch.split(y2, y2.size()[-1] // tp_size, dim=-1)
        assert np.allclose(y1.detach().cpu().numpy(), torch_out_list[tp_rank].detach().cpu().numpy())

    weight_grad_list = weight_grad2.chunk(tp_size, dim=0)
    assert np.allclose(weight_grad1.reshape(weight_grad_list[tp_rank].shape).cpu().numpy(), weight_grad_list[tp_rank].cpu().numpy())

    bias_grad_list = bias_grad2.chunk(tp_size, dim=0)
    assert np.allclose(bias_grad1.reshape(bias_grad_list[tp_rank].shape).cpu().numpy(), bias_grad_list[tp_rank].cpu().numpy())

    if gather_input:
        x_grad_list = x.grad.chunk(tp_size, dim=0)
        np.testing.assert_allclose(rank_x.grad.cpu().numpy(), x_grad_list[tp_rank].cpu().numpy(), atol=1e-4, rtol=1e-4)
    else:
        np.testing.assert_allclose(rank_x.grad.cpu().numpy(), x.grad.cpu().numpy(), atol=1e-4, rtol=1e-4)

def test_gather_output():
    run(True, True, 'linear.ckp')

def test_no_gather_output():
    run(True, False, 'linear_no_gather.ckp')

def test_no_gather_input():
    run(False, True, 'linear.ckp')


if __name__ == "__main__":
    bmt.init_distributed(tp_size=2)
    test_gather_output()
    test_no_gather_output()
    test_no_gather_input()

