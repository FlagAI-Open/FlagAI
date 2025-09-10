import torch
import bmtrain as bmt

from bmtrain.global_var import config
from bmtrain import nccl, distributed
from bmtrain.synchronize import gather_result

def test_main():

    ref_result = torch.ones(5 * bmt.world_size(), 5)  
    tensor = ref_result.chunk(bmt.world_size(), dim=0)[bmt.rank()]
    real_result = bmt.gather_result(tensor)
    assert torch.allclose(ref_result, real_result, atol=1e-6), "Assertion failed for real gather result error"

    for i in range(4):
        size = i + 1  
        tensor_slice = tensor[:size, :size]
        result_slice = bmt.gather_result(tensor_slice)
        test_slice = torch.chunk(result_slice, bmt.world_size(), dim=0)[i]
        assert torch.allclose(tensor_slice, test_slice), f"Assertion failed for tensor_slice_{i}"

print("All test passed")

if __name__ == '__main__':
    bmt.init_distributed(pipe_size=1)
    test_main()
