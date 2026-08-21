import torch
import bmtrain as bmt

class Layer(torch.nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.linear = bmt.nn.Linear(32, 32)
        self.count = 0
    def forward(self, x):
        self.count += 1
        return self.linear(x)

def test_no_grad():
    x = torch.randn(32, 32, device='cuda')

    layer1 = bmt.Block(Layer())
    layer2 = bmt.Block(Layer())
    layer1.linear.weight.requires_grad_(False)
    layer1.linear.bias.requires_grad_(False)
    y = layer1(x)
    assert y.requires_grad == False
    y = layer2(y)
    y.sum().backward()
    assert layer1.count == 1
    assert layer2.count == 2

def test_multi_layer_no_grad():
    x = torch.randn(32, 32, device='cuda')

    layers = []
    for i in range(10):
        layer = bmt.Block(Layer())
        layer.linear.weight.requires_grad_(i > 4)
        layer.linear.bias.requires_grad_(i > 4)
        layers.append(layer)

    y = x
    for layer in layers:
        y = layer(y)
    y.sum().backward()
    for i in range(len(layers)):
        assert layers[i].count == (1 if i <=4 else 2)

def test_all_input_no_grad():
    linear1 = bmt.nn.Linear(32, 32)
    linear2 = bmt.nn.Linear(32, 32)

    x = torch.randn(32,32, device='cuda')

    linear1 = bmt.Block(linear1)
    linear2 = bmt.Block(linear2)
    y = linear1(x)
    y = linear2(y)
    y.sum().backward()
    assert linear1.weight.grad is not None
    assert linear1.bias.grad is not None
    assert x.grad is None

def test_same_layer():
    layer = Layer()
    block_list = bmt.TransformerBlockList([layer, layer])
    assert id(block_list[0]) != id(block_list[1])

def test_no_grad_error():
    layer = bmt.Block(Layer())

    try:
        block_list = bmt.TransformerBlockList([layer, layer])
        raise ValueError("test failed")
    except AssertionError as e:
        return

def test_no_grad_error2():
    layer = bmt.Block(Layer())

    try:
        block_list = bmt.PipelineTransformerBlockList([layer])
        raise ValueError("test failed")
    except AssertionError as e:
        return

if __name__ == '__main__':
    bmt.init_distributed()

    test_no_grad()
    test_multi_layer_no_grad()
    test_all_input_no_grad()
    test_same_layer()
    test_no_grad_error()
    test_no_grad_error2()
