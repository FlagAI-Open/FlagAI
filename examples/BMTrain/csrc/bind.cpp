#include "include/bind.hpp"

PYBIND11_MODULE(C, m) {
    m.def("to_fp16_from_fp32", &fp16_from_fp32_value_launcher, "convert");
    m.def("to_bf16_from_fp32", &bf16_from_fp32_value_launcher, "convert");
    m.def("is_bf16_supported", &is_bf16_supported, "whether bf16 supported");
    m.def("has_nan_inf_fp16_launcher", &has_nan_inf_fp16_launcher, "has nan inf");
    m.def("has_nan_inf_bf16_launcher", &has_nan_inf_bf16_launcher, "has nan inf bf16");
    m.def("adam_fp16_launcher", &adam_fp16_launcher, "adam function cpu");
    m.def("adam_bf16_launcher", &adam_bf16_launcher, "adam function cpu");
    m.def("adam_cpu_fp16_launcher", &adam_cpu_fp16_launcher, "adam function cpu");
    m.def("adam_cpu_bf16_launcher", &adam_cpu_bf16_launcher, "adam function cpu");
    m.def("cross_entropy_forward_fp16_launcher", &cross_entropy_forward_fp16_launcher, "cross entropy forward");
    m.def("cross_entropy_forward_bf16_launcher", &cross_entropy_forward_bf16_launcher, "cross entropy forward");
    m.def("cross_entropy_backward_inplace_fp16_launcher", &cross_entropy_backward_inplace_fp16_launcher, "cross entropy backward inplace");
    m.def("cross_entropy_backward_inplace_bf16_launcher", &cross_entropy_backward_inplace_bf16_launcher, "cross entropy backward inplace");
    m.def("fused_sumexp_fp16_launcher", &fused_sumexp_fp16_launcher, "sum exp");
    m.def("fused_sumexp_bf16_launcher", &fused_sumexp_bf16_launcher, "sum exp");
    m.def("fused_softmax_inplace_fp16_launcher", &fused_softmax_inplace_fp16_launcher, "softmax inplace");
    m.def("fused_softmax_inplace_bf16_launcher", &fused_softmax_inplace_bf16_launcher, "softmax inplace");
    m.def("ncclGetUniqueId", &pyNCCLGetUniqueID, "nccl get unique ID");
    m.def("ncclCommInitRank", &pyNCCLCommInitRank, "nccl init rank");
    m.def("ncclCommDestroy", &pyNCCLCommDestroy, "nccl delete rank");
    m.def("ncclAllGather", &pyNCCLAllGather, "nccl all gather");
    m.def("ncclAllReduce", &pyNCCLAllReduce, "nccl all reduce");
    m.def("ncclBroadcast", &pyNCCLBroadcast, "nccl broadcast");
    m.def("ncclReduce", &pyNCCLReduce, "nccl reduce");
    m.def("ncclReduceScatter", &pyNCCLReduceScatter, "nccl reduce scatter");
    m.def("ncclGroupStart", &pyNCCLGroupStart, "nccl group start");
    m.def("ncclGroupEnd", &pyNCCLGroupEnd, "nccl group end");
    m.def("ncclSend", &pyNCCLSend, "nccl send");
    m.def("ncclRecv", &pyNCCLRecv, "nccl recv");
    m.def("ncclCommCount", &pyNCCLCommCount, "nccl comm count");
    m.def("ncclCommUserRank", &pyNCCLCommUserRank, "nccl comm user rank");
}
