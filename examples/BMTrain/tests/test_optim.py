from utils import *
import torch
import bmtrain as bmt
from bmtrain import optim

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128, bias=False)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 128)
        self.param = torch.nn.Parameter(torch.empty(1237))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

def main(dtype):
    model1 = TestModule()
    model2 = TestModule()
    model3 = TestModule()
    model4 = TestModule()
    model5 = TestModule()

    state_dict = model1.state_dict()
    for kw in state_dict.keys():
        state_dict[kw] = torch.randn_like(state_dict[kw])

    model1.load_state_dict(state_dict)
    model2.load_state_dict(state_dict)
    model3.load_state_dict(state_dict)
    model4.load_state_dict(state_dict)
    model5.load_state_dict(state_dict)

    model1 = model1.cuda().to(dtype)
    model2 = model2.cuda().to(dtype)
    model3 = model3.cuda()
    model4 = model4.cuda()
    model5 = model5.cuda()

    opt1 = bmt.optim.AdamOptimizer(model1.parameters(), lr=1)
    opt2 = bmt.optim.AdamOffloadOptimizer(model2.parameters(), lr=1)
    opt3 = torch.optim.Adam(model3.parameters(), lr=1)
    opt4 = bmt.optim.AdamOptimizer(model4.parameters(), lr=1)
    opt5 = bmt.optim.AdamOffloadOptimizer(model5.parameters(), lr=1)

    optim_manager = bmt.optim.OptimManager(loss_scale=4)
    optim_manager.add_optimizer(opt1)
    optim_manager.add_optimizer(opt2)
    optim_manager.add_optimizer(opt3)
    optim_manager.add_optimizer(opt4)
    optim_manager.add_optimizer(opt5)

    for _ in range(100):
        optim_manager.zero_grad()

        for p1, p2, p3, p4, p5 in zip(model1.parameters(), model2.parameters(), model3.parameters(), model4.parameters(), model5.parameters()):
            grad = torch.randn_like(p1)
            p1.grad = grad.to(dtype)
            p2.grad = grad.to(dtype)
            p3.grad = grad.float()
            p4.grad = grad.float()
            p5.grad = grad.float()

        optim_manager.step()
        torch.cuda.synchronize()

        for p1, p2, p3, p4, p5 in zip(model1.parameters(), model2.parameters(), model3.parameters(), model4.parameters(), model5.parameters()):
            diff1 = torch.abs(p1 - p2).max().item() 
            diff2 = torch.abs(p1 - p3).max().item()
            diff3 = torch.abs(p2 - p3).max().item()
            diff4 = torch.abs(p3 - p4).max().item()
            diff5 = torch.abs(p3 - p5).max().item()
            print(f"{diff1:.6f}, {diff2:.6f}, {diff3:.6f}, {diff4:.6f}, {diff5:.6f}")
            assert_lt(diff1, 1)
            assert_lt(diff2, 1)
            assert_lt(diff3, 1)
            assert_eq(diff4, 0)
            assert_lt(diff5, 0.00001)

if __name__ == "__main__":
    bmt.init_distributed()
    main(torch.float16)
    print("==============================================================================")
    try:
        main(torch.bfloat16)
    except NotImplementedError: 
        pass
