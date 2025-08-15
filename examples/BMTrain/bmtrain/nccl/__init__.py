
from typing_extensions import Literal
import torch
from .. import C
from .enums import *

class NCCLCommunicator:
    """
    NCCL communicator stores the communicator handle.
    """

    def __init__(self, ptr) -> None:
        self.__ptr = ptr
    
    @property
    def ptr(self):
        """
        Returns the communicator handle.
        """
        if self.__ptr == -1:
            raise RuntimeError("NCCL Communicator is already destroyed")
        return self.__ptr
    
    def _destroy_ptr(self):
        self.__ptr = -1

# utils

def dtype2nccl(dtype : torch.dtype) -> int:
    MAP = {
        torch.int8: ncclInt8,
        torch.uint8 : ncclUint8,
        torch.int32 : ncclInt32,
        torch.int : ncclInt32,
        torch.int64 : ncclInt64,
        torch.float16 : ncclFloat16,
        torch.half : ncclHalf,
        torch.bfloat16 : ncclBFloat16,
        torch.float32 : ncclFloat32,
        torch.float : ncclFloat,
        torch.float64 : ncclFloat64,
        torch.double : ncclDouble,
        torch.bool : ncclBool
    }
    if dtype not in MAP:
        raise TypeError("Unsupport dtype %s" % dtype)
    return MAP[dtype]

def op2nccl(
    op : Literal["sum", "prod", "max", "min", "avg"]
):
    if op == "sum":
        return ncclSum
    if op == "prod":
        return ncclProd
    if op == "max":
        return ncclMax
    if op == "min":
        return ncclMin
    if op == "avg":
        return ncclAvg
    raise ValueError("Unknown gather op %s")

# wrappers

def getUniqueId() -> bytes:
    """
    NCCL API: `ncclGetUniqueId <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclgetuniqueid>`_

    """
    return C.ncclGetUniqueId()

def commInitRank(unique_id : bytes, world_size : int, rank : int) -> NCCLCommunicator:
    """
    NCCL API: `ncclCommInitRank <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrank>`_

    """
    assert rank >= 0 and rank < world_size, "rank must be between 0 and world_size-1"
    return NCCLCommunicator(C.ncclCommInitRank(unique_id, world_size, rank))

def commDestroy(comm : NCCLCommunicator):
    """
    NCCL API: `ncclCommDestroy <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy>`_

    """
    C.ncclCommDestroy(comm.ptr)
    comm._destroy_ptr()
def commCount(comm : NCCLCommunicator):
    """NCCL API: `ncclCommCount <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommcount>`_

    Args:
        comm (NCCLCommunicator): NCCL communicator.
    """
    return C.ncclCommCount(comm.ptr)
### collective
def commRank(comm : NCCLCommunicator):
    """NCCL API: `ncclCommUserRank <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclCommUserRank>`_

    Args:
        comm (NCCLCommunicator): NCCL communicator.
    """
    return C.ncclCommUserRank(comm.ptr)
def allReduce(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        op : Literal["sum", "prod", "max", "min", "avg"],
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclAllReduce <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        op (Literal["sum", "prod", "max", "min", "avg"]): Reduction operation.
        comm (NCCLCommunicator): NCCL communicator.
    
    The src and dst buffers must be the same size, type and on the same device.

    If src == dst, the operation is performed in-place.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    count = src.size()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    assert src.size() == dst.size(), "Buffer size not aligned"
    C.ncclAllReduce(
        sendbuff,
        recvbuff,
        count,
        datatype,
        operator,
        comm.ptr,
        torch.cuda.current_stream().cuda_stream
    )
def send(src : torch.storage._StorageBase,
         peer : int,
         comm : NCCLCommunicator
    ):
    """NCCL API: `ncclsend <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend>`_

        Args:
            src (torch.storage._StorageBase): Source buffer.
            peer (int): rank peer needs to call ncclRecv
            comm (NCCLCommunicator): NCCL communicator.
    """

    sendbuff = src.data_ptr()
    count = src.size()
    datatype = dtype2nccl(src.dtype)
    C.ncclSend(
        sendbuff,
        count,
        datatype,
        peer,
        comm.ptr,
        torch.cuda.current_stream().cuda_stream
    )
def recv(dst : torch.storage._StorageBase,
         peer : int,
         comm : NCCLCommunicator
        ):
    recvbuff = dst.data_ptr()
    count = dst.size()
    datatype = dtype2nccl(dst.dtype)
    C.ncclRecv(
        recvbuff,
        count,
        datatype,
        peer,
        comm.ptr,
        torch.cuda.current_stream().cuda_stream
    )
    
def broadcast(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        root : int,
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclBroadcast <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        root (int): Rank of the root.
        comm (NCCLCommunicator): NCCL communicator.
    
    The src and dst buffers must be the same size, type and on the same device.

    If src == dst, the operation is performed in-place.

    """

    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    count = src.size()
    datatype = dtype2nccl(src.dtype)

    assert dst.size() == src.size(), "Buffer size not aligned"
    C.ncclBroadcast(
        sendbuff, 
        recvbuff, 
        count, 
        datatype, 
        root, 
        comm.ptr, 
        torch.cuda.current_stream().cuda_stream
    )

def reduce(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        op : Literal["sum", "prod", "max", "min", "avg"],
        root : int,
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclReduce <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreduce>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        op (Literal["sum", "prod", "max", "min", "avg"]): Reduction operation.
        root (int): Rank of the root.
        comm (NCCLCommunicator): NCCL communicator.
    
    The src and dst buffers must be the same size, type and on the same device.

    If src == dst, the operation is performed in-place.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    count = src.size()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    assert dst.size() == src.size(), "Buffer size not aligned"
    C.ncclReduce(sendbuff, recvbuff, count, datatype, operator, root, comm.ptr, torch.cuda.current_stream().cuda_stream)

def allGather(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclAllGather <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        comm (NCCLCommunicator): NCCL communicator.
    
    The size of the dst buffer must be equal to the size of src buffer * world_size.

    The dst buffer is only used on rank root.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    sendcount = src.size()
    datatype = dtype2nccl(src.dtype)
    assert dst.size() % sendcount == 0, "Buffer size not aligned"
    C.ncclAllGather(
        sendbuff, 
        recvbuff, 
        sendcount, 
        datatype, 
        comm.ptr, 
        torch.cuda.current_stream().cuda_stream
    )


def reduceScatter(
        src : torch.storage._StorageBase,
        dst : torch.storage._StorageBase,
        op : Literal["sum", "prod", "max", "min", "avg"],
        comm : NCCLCommunicator
    ):
    """NCCL API: `ncclReduceScatter <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter>`_

    Args:
        src (torch.storage._StorageBase): Source buffer.
        dst (torch.storage._StorageBase): Destination buffer.
        op (Literal["sum", "prod", "max", "min", "avg"]): Reduction operation.
        comm (NCCLCommunicator): NCCL communicator.
    
    The size of the dst buffer must be equal to the size of src buffer / world_size.

    The dst buffer on rank `i` will contail the i-th block of the reduced result.

    """
    assert src.dtype == dst.dtype, "send and recv buffers must be the same time"
    assert src.is_cuda and dst.is_cuda

    sendbuff = src.data_ptr()
    recvbuff = dst.data_ptr()
    recvcount = dst.size()
    datatype = dtype2nccl(src.dtype)
    operator = op2nccl(op)

    assert src.size() % recvcount == 0, "Buffer size not aligned"
    C.ncclReduceScatter(
        sendbuff,
        recvbuff,
        recvcount,
        datatype,
        operator,
        comm.ptr,
        torch.cuda.current_stream().cuda_stream
    )

def groupStart():
    """
    NCCL API: `ncclGroupStart <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupstart>`_
    """
    C.ncclGroupStart()

def groupEnd():
    """
    NCCL API: `ncclGroupEnd <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupend>`_
    """
    C.ncclGroupEnd()
