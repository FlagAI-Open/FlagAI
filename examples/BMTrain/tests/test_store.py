import bmtrain as bmt
from bmtrain.store import allgather_object

def test_allgather_object():

    res = allgather_object(bmt.rank(), bmt.config["comm"])
    ref = [i for i in range(bmt.world_size())]
    assert res == ref

if __name__ == "__main__":
    bmt.init_distributed()
    test_allgather_object()

