from utils import *

import torch
import bmtrain as bmt
from bmtrain.global_var import config

def test_send_recv():
    if config["topology"].stage_id == 0:
        a = torch.ones((2,1)) * (config["zero_rank"]+1)
        a = a.cuda()
        print(f"send {a}")
        bmt.distributed.send_activations(a, 1, config["pipe_comm"])
    else:
        ref = torch.ones((2,1)) * (config["zero_rank"]+1)
        a = bmt.distributed.recv_activations(0, config["pipe_comm"])
        print(f"recv {a}")
        assert_all_eq(a, ref.cuda())

if __name__ == '__main__':
    bmt.init_distributed(pipe_size=2)

    test_send_recv()