import torch
import bmtrain as bmt
import os
from copy import deepcopy
from bmtrain import optim, lr_scheduler

class TestSubModule(bmt.DistributedModule):
    def __init__(self):
        super(TestSubModule, self).__init__()
        self.fc1 = bmt.BMTrainModelWrapper(torch.nn.Linear(768, 3072))
        self.fc2 = bmt.BMTrainModelWrapper(torch.nn.Linear(3072, 1024))
        self.fc3 = bmt.BMTrainModelWrapper(torch.nn.Linear(1024, 768))
        self.param = bmt.DistributedParameter(torch.zeros(1237))
        self.fc4 = bmt.BMTrainModelWrapper(torch.nn.Linear(768, 300))
        self.fc5 = bmt.BMTrainModelWrapper(torch.nn.Linear(300, 768))
        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.layer1 = TestSubModule()
        self.layer2 = bmt.CheckpointBlock(TestSubModule())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def train(model1, model2, model3, optim_manager):
    x = torch.randn((4, 768)).cuda()
    for _ in range(10):
        optim_manager.zero_grad()

        y1, y2, y3 = model1(x), model2(x), model3(x)
        w = torch.randn_like(y1)
        loss = (y1*w).sum() + (y2*w).sum() + (y3*w).sum()
        optim_manager.backward(loss)
        
        optim_manager.step()

def manual_seed(seed=33):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

def main():
    model1 = TestModule()
    model2 = TestModule()
    model3 = TestModule()

    bmt.save(model1, "test_optim_state_model1.pt")

    bmt.load(model1, f"test_optim_state_model1.pt")
    bmt.load(model2, f"test_optim_state_model1.pt")
    bmt.load(model3, f"test_optim_state_model1.pt")

    opt1 = optim.AdamOptimizer(model1.parameters(), weight_decay=1e-3)
    opt2 = optim.AdamOffloadOptimizer(model2.parameters(), weight_decay=1e-3)
    opt3 = torch.optim.Adam(model3.parameters(), weight_decay=1e-3)
    lrs1 = lr_scheduler.Noam(opt1, start_lr=20, warmup_iter=0, end_iter=300, num_iter=1)
    lrs2 = lr_scheduler.Noam(opt2, start_lr=20, warmup_iter=0, end_iter=300, num_iter=1)
    lrs3 = lr_scheduler.Noam(opt3, start_lr=20, warmup_iter=0, end_iter=300, num_iter=1)
    optim_manager = optim.OptimManager(loss_scale=256)
    optim_manager.add_optimizer(opt1, lrs1)
    optim_manager.add_optimizer(opt2, lrs2)
    optim_manager.add_optimizer(opt3, lrs3)

    train(model1, model2, model3, optim_manager)

    bmt.save(model1, f"test_optim_state_model1.pt")
    bmt.save(model2, f"test_optim_state_model2.pt")
    bmt.save(model3, f"test_optim_state_model3.pt")

    torch.save(optim_manager.state_dict(), f"test_optim_manager_{bmt.rank()}.opt")

    manual_seed()
    train(model1, model2, model3, optim_manager)
    state_2 = deepcopy([list(model1.parameters()), list(model2.parameters()), list(model3.parameters())])

    model1 = TestModule()
    model2 = TestModule()
    model3 = TestModule()
    bmt.load(model1, f"test_optim_state_model1.pt")
    bmt.load(model2, f"test_optim_state_model2.pt")
    bmt.load(model3, f"test_optim_state_model3.pt")

    opt1 = optim.AdamOptimizer(model1.parameters(), weight_decay=1e-8, betas=(0.3, 0.333), eps=1e-3)
    opt2 = optim.AdamOffloadOptimizer(model2.parameters(), weight_decay=1e-7, betas=(0.4, 0.456), eps=1e-1)
    opt3 = torch.optim.Adam(model3.parameters(), weight_decay=1e-6, betas=(0.9, 0.777), eps=1e-2)
    lrs1 = lr_scheduler.Noam(opt1, start_lr=200, warmup_iter=30, end_iter=500, num_iter=3)
    lrs2 = lr_scheduler.Noam(opt2, start_lr=20, warmup_iter=40, end_iter=600, num_iter=1)
    lrs3 = lr_scheduler.Noam(opt3, start_lr=10, warmup_iter=50, end_iter=700, num_iter=2)
    optim_manager = optim.OptimManager(loss_scale=10485760)
    optim_manager.add_optimizer(opt1, lrs1)
    optim_manager.add_optimizer(opt2, lrs2)
    optim_manager.add_optimizer(opt3, lrs3)
    optim_manager.load_state_dict(torch.load(f"test_optim_manager_{bmt.rank()}.opt"))

    manual_seed()
    train(model1, model2, model3, optim_manager)
    state_1_plus_1 = deepcopy([list(model1.parameters()), list(model2.parameters()), list(model3.parameters())])

    for i, kind in [
        (0, "BMTAdam"),
        (1, "BMTAdamOffload"),
        (2, "TorchAdam")
    ]:
        ref = state_2[i]
        chk = state_1_plus_1[i]
        for rp, p in zip(ref, chk):
            assert (rp==p).all(), f"{kind} state load error"

    if bmt.rank() == 0:
        os.remove(f"test_optim_state_model1.pt")
        os.remove(f"test_optim_state_model2.pt")
        os.remove(f"test_optim_state_model3.pt")
    os.remove(f"test_optim_manager_{bmt.rank()}.opt")

if __name__ == "__main__":
    bmt.init_distributed()
    main()
