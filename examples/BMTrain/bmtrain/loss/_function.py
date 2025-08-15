from .. import C
import torch

CHECK_INPUT = lambda x: x.is_contiguous() and x.is_cuda


def has_inf_nan(g_half: torch.Tensor, out: torch.Tensor) -> None:
    assert out.dtype == torch.uint8, "out must be a uint8 tensor"
    assert CHECK_INPUT(g_half), "g_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(out), "out must be contiguous and on cuda"
    mid = torch.zeros(1024, device=out.device, dtype=out.dtype)
    stream = torch.cuda.current_stream().cuda_stream
    if g_half.dtype == torch.float16:
        C.has_nan_inf_fp16_launcher(
            g_half.numel(), g_half.data_ptr(), mid.data_ptr(), out.data_ptr(), stream
        )
    elif g_half.dtype == torch.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.has_nan_inf_bf16_launcher(
            g_half.numel(), g_half.data_ptr(), mid.data_ptr(), out.data_ptr(), stream
        )
    else:
        raise ValueError(f"has_inf_nan not supported for dtype {g_half.dtype}")


def cross_entropy_forward(
    m: int,
    n: int,
    input: torch.Tensor,
    target: torch.Tensor,
    softmax: torch.Tensor,
    output: torch.Tensor,
    ignore_index: int,
) -> None:
    CHECK_INPUT(input)
    CHECK_INPUT(target)
    CHECK_INPUT(softmax)
    CHECK_INPUT(output)
    assert target.dtype == torch.int32, "target must be an int tensor"
    assert output.dtype == torch.float32, "output must be a float tensor"
    assert (
        input.numel() == softmax.numel()
    ), "input and softmax must have the same number of elements"
    assert (
        target.numel() == output.numel()
    ), "target and output must have the same number of elements"
    input_ptr = input.data_ptr()
    target_ptr = target.data_ptr()
    softmax_ptr = softmax.data_ptr()
    output_ptr = output.data_ptr()
    cuda_stream = torch.cuda.current_stream().cuda_stream
    if input.dtype == torch.float16:
        C.cross_entropy_forward_fp16_launcher(
            m,
            n,
            input_ptr,
            target_ptr,
            softmax_ptr,
            output_ptr,
            ignore_index,
            cuda_stream,
        )
    elif input.dtype == torch.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.cross_entropy_forward_bf16_launcher(
            m,
            n,
            input_ptr,
            target_ptr,
            softmax_ptr,
            output_ptr,
            ignore_index,
            cuda_stream,
        )
    else:
        raise ValueError(f"cross_entropy_forward not supported for dtype {input.dtype}")


def cross_entropy_backward_inplace(
    m: int,
    n: int,
    grad_output: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    ignore_index: int,
) -> None:
    CHECK_INPUT(grad_output)
    CHECK_INPUT(target)
    CHECK_INPUT(x)
    assert grad_output.dtype == torch.float32, "grad_output must be a float tensor"
    assert target.dtype == torch.int32, "target must be an int tensor"
    assert (
        target.numel() == grad_output.numel()
    ), "target and grad_output must have the same number of elements"
    cuda_stream = torch.cuda.current_stream().cuda_stream
    grad_output_ptr = grad_output.data_ptr()
    target_ptr = target.data_ptr()
    x_ptr = x.data_ptr()

    if x.dtype == torch.float16:
        C.cross_entropy_backward_inplace_fp16_launcher(
            m, n, grad_output_ptr, target_ptr, x_ptr, ignore_index, cuda_stream
        )
    elif x.dtype == torch.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.cross_entropy_backward_inplace_bf16_launcher(
            m, n, grad_output_ptr, target_ptr, x_ptr, ignore_index, cuda_stream
        )
    else:
        raise ValueError(
            f"cross_entropy_backward not supported for dtype {input.dtype}"
        )


def fused_sumexp(logits: torch.Tensor, max_logits: torch.Tensor) -> torch.Tensor:
    CHECK_INPUT(logits)
    CHECK_INPUT(max_logits)
    assert max_logits.dtype == torch.float32, "max_logits must be float tensor"
    assert max_logits.size(0) == logits.size(
        0
    ), "max_logits must have same size(0) as logits"
    sum_exp_logits = torch.empty(
        logits.size(0), dtype=torch.float32, device=logits.device
    )
    m = logits.size(0)
    n = logits.size(1)
    cuda_stream = torch.cuda.current_stream().cuda_stream
    logits_ptr = logits.data_ptr()
    max_logits_ptr = max_logits.data_ptr()
    sum_exp_logits_ptr = sum_exp_logits.data_ptr()
    if logits.dtype == torch.float16:
        C.fused_sumexp_fp16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    elif logits.dtype == torch.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.fused_sumexp_bf16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    else:
        raise ValueError(f"fused_sumexp not supported for dtype {logits.dtype}")
    return sum_exp_logits


def fused_softmax_inplace(
    logits: torch.Tensor, max_logits: torch.Tensor, sum_exp_logits: torch.Tensor
) -> None:
    CHECK_INPUT(logits)
    CHECK_INPUT(max_logits)
    CHECK_INPUT(sum_exp_logits)
    assert max_logits.dtype == torch.float32, "max_logits must be float tensor"
    assert sum_exp_logits.dtype == torch.float32, "sum_exp_logits must be float tensor"
    assert max_logits.size(0) == logits.size(
        0
    ), "max_logits must have same size(0) as logits"
    assert sum_exp_logits.size(0) == logits.size(
        0
    ), "sum_exp_logits must have same size(0) as logits"
    m = logits.size(0)
    n = logits.size(1)
    cuda_stream = torch.cuda.current_stream().cuda_stream
    logits_ptr = logits.data_ptr()
    max_logits_ptr = max_logits.data_ptr()
    sum_exp_logits_ptr = sum_exp_logits.data_ptr()
    if logits.dtype == torch.float16:
        C.fused_softmax_inplace_fp16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    elif logits.dtype == torch.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        C.fused_softmax_inplace_bf16_launcher(
            m, n, logits_ptr, max_logits_ptr, sum_exp_logits_ptr, cuda_stream
        )
    else:
        raise ValueError(
            f"fused_softmax_inplace not supported for dtype {logits.dtype}"
        )
