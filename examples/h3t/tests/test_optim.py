import torch
import bmtrain as bmt

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 128)
        self.param = torch.nn.Parameter(torch.empty(1237))

def main():
    # FIXME: this test script is not working
    model1 = TestModule()
    model2 = TestModule()
    model3 = TestModule()

    state_dict = model1.state_dict()
    for kw in state_dict.keys():
        state_dict[kw] = torch.randn_like(state_dict[kw])
    
    model1.load_state_dict(state_dict)
    model2.load_state_dict(state_dict)
    model3.load_state_dict(state_dict)

    model1 = model1.cuda().half()
    model2 = model2.cuda().half()
    model3 = model3.cuda()
    
    opt1 = bmt.optim.AdamOptimizer(model1.parameters(), weight_decay=1e-3)
    opt2 = bmt.optim.AdamOffloadOptimizer(model2.parameters(), weight_decay=1e-3)
    opt3 = torch.optim.Adam(model3.parameters(), weight_decay=1e-3)

    for _ in range(100):
        opt1.zero_grad()
        opt2.zero_grad()
        opt3.zero_grad()

        for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
            grad = torch.randn_like(p1)
            p1.grad = grad
            p2.grad = grad
            p3.grad = grad.float()
        
        opt1.step()
        opt2.step()
        opt3.step()

        for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
            diff1 = torch.abs(p1 - p2).max()
            diff2 = torch.abs(p1 - p3).max()
            diff3 = torch.abs(p2 - p3).max()
            print(diff1, diff2, diff3)

if __name__ == "__main__":
    main()
