from .. import C
import torch

CHECK_INPUT = lambda x: x.is_contiguous() and x.is_cuda


def bf16_from_fp32(param_fp32):
    param_bf16 = torch.empty_like(param_fp32, dtype=torch.bfloat16)
    C.to_bf16_from_fp32(
        param_fp32.numel(), param_fp32.data_ptr(), param_bf16.data_ptr()
    )
    return param_bf16


def fp16_from_fp32(param_fp32):
    param_fp16 = torch.empty_like(param_fp32, dtype=torch.float16)
    C.to_fp16_from_fp32(
        param_fp32.numel(), param_fp32.data_ptr(), param_fp16.data_ptr()
    )
    return param_fp16


def adam_cpu(
    param_fp32: torch.Tensor,
    param_fp16: torch.Tensor,
    delta_info: torch.Tensor,
    g_fp16: torch.Tensor,
    m_fp32: torch.Tensor,
    v_fp32: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    scale: float,
    weight_decay: float,
    step: int,
) -> None:
    assert param_fp32.is_contiguous(), "param_fp32 must be contiguous"
    assert param_fp16.is_contiguous(), "param_fp16 must be contiguous"
    assert g_fp16.is_contiguous(), "g_fp16 must be contiguous"
    assert m_fp32.is_contiguous(), "m_fp32 must be contiguous"
    assert v_fp32.is_contiguous(), "v_fp32 must be contiguous"
    assert param_fp32.dtype == torch.float32, "param_fp32 must be float32 tensor"
    assert (
        param_fp16.dtype == torch.float16 or param_fp16.dtype == torch.bfloat16
    ), "param_fp16 must be float16/bfloat16 tensor"
    assert (
        g_fp16.dtype == torch.float16 or g_fp16.dtype == torch.bfloat16
    ), "g_fp16 must be float16/bfloat16 tensor"
    assert m_fp32.dtype == torch.float32, "m_fp32 must be float32 tensor"
    assert v_fp32.dtype == torch.float32, "v_fp32 must be float32 tensor"
    assert param_fp32.device == torch.device("cpu"), "param_fp32 must be a cpu tensor"
    assert param_fp16.device == torch.device("cpu"), "param_fp16 must be a cpu tensor"
    assert g_fp16.device == torch.device("cpu"), "g_fp16 must be a cpu tensor"
    assert m_fp32.device == torch.device("cpu"), "m_fp32 must be a cpu tensor"
    assert v_fp32.device == torch.device("cpu"), "v_fp32 must be a cpu tensor"
    assert (
        param_fp32.numel() == param_fp16.numel()
    ), "param_fp32 and param_fp16 must have the same number of elements"
    assert (
        param_fp32.numel() == g_fp16.numel()
    ), "param_fp32 and g_fp16 must have the same number of elements"
    assert (
        param_fp32.numel() == m_fp32.numel()
    ), "param_fp32 and m_fp32 must have the same number of elements"
    assert (
        param_fp32.numel() == v_fp32.numel()
    ), "param_fp32 and v_fp32 must have the same number of elements"
    if delta_info is not None:
        assert delta_info.is_contiguous(), "delta_info must be contiguous"
        assert delta_info.dtype == torch.float32, "delta_info must be float32 tensor"
        assert delta_info.device == torch.device(
            "cpu"
        ), "delta_info must be a cpu tensor"
        assert delta_info.numel() == 4, "delta_info have a length of 4"
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    if g_fp16.dtype == torch.float16:
        launcher = C.adam_cpu_fp16_launcher
    elif g_fp16.dtype == torch.bfloat16:
        if not C.is_bf16_supported():
            raise NotImplementedError(f"bfloat16 is not supported on current GPU")
        launcher = C.adam_cpu_bf16_launcher
    launcher(
        param_fp32.numel(),
        param_fp32.data_ptr(),
        param_fp16.data_ptr(),
        delta_info.data_ptr() if delta_info is not None else 0,
        g_fp16.data_ptr(),
        m_fp32.data_ptr(),
        v_fp32.data_ptr(),
        beta1,
        beta2,
        eps,
        lr,
        scale,
        weight_decay,
        bias_correction1,
        bias_correction2,
    )


def adam_fp16(
    param_fp32: torch.Tensor,
    param_fp16: torch.Tensor,
    g_fp16: torch.Tensor,
    m_fp16: torch.Tensor,
    v_fp32: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    scale: float,
    weight_decay: float,
    step: int,
) -> None:
    assert CHECK_INPUT(param_fp32), "param_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(param_fp16), "param_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(g_fp16), "g_fp16 must be contiguous and on cuda"
    assert CHECK_INPUT(m_fp16), "m_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(v_fp32), "v_fp32 must be contiguous and on cuda"
    assert param_fp32.dtype == torch.float32, "param_fp32 must be float32 tensor"
    assert param_fp16.dtype == torch.float16, "param_fp16 must be float16 tensor"
    assert g_fp16.dtype == torch.float16, "g_fp16 must be float16 tensor"
    assert m_fp16.dtype == torch.float16, "m_fp16 must be float16 tensor"
    assert v_fp32.dtype == torch.float32, "v_fp32 must be float32 tensor"
    assert (
        param_fp32.numel() == param_fp16.numel()
    ), "param_fp32 and param_fp16 must have the same number of elements"
    assert (
        param_fp32.numel() == g_fp16.numel()
    ), "param_fp32 and g_fp16 must have the same number of elements"
    assert (
        param_fp32.numel() == m_fp16.numel()
    ), "param_fp32 and m_fp32 must have the same number of elements"
    assert (
        param_fp32.numel() == v_fp32.numel()
    ), "param_fp32 and v_fp32 must have the same number of elements"
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    stream = torch.cuda.current_stream().cuda_stream
    C.adam_fp16_launcher(
        param_fp32.numel(),
        param_fp32.data_ptr(),
        param_fp16.data_ptr(),
        g_fp16.data_ptr(),
        m_fp16.data_ptr(),
        v_fp32.data_ptr(),
        beta1,
        beta2,
        eps,
        lr,
        scale,
        weight_decay,
        bias_correction1,
        bias_correction2,
        stream,
    )


def adam_bf16(
    param_fp32: torch.Tensor,
    param_bf16: torch.Tensor,
    g_bf16: torch.Tensor,
    m_fp32: torch.Tensor,
    v_fp32: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    lr: float,
    scale: float,
    weight_decay: float,
    step: int,
) -> None:
    assert CHECK_INPUT(param_fp32), "param_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(param_bf16), "param_bf16 must be contiguous and on cuda"
    assert CHECK_INPUT(g_bf16), "g_bf16 must be contiguous and on cuda"
    assert CHECK_INPUT(m_fp32), "m_fp32 must be contiguous and on cuda"
    assert CHECK_INPUT(v_fp32), "v_fp32 must be contiguous and on cuda"
    assert param_fp32.dtype == torch.float32, "param_fp32 must be float32 tensor"
    assert param_bf16.dtype == torch.bfloat16, "param_fp16 must be float16 tensor"
    assert g_bf16.dtype == torch.bfloat16, "g_bf16 must be bfloat16 tensor"
    assert m_fp32.dtype == torch.float32, "m_fp32 must be bfloat16 tensor"
    assert v_fp32.dtype == torch.float32, "v_fp32 must be float32 tensor"
    assert (
        param_fp32.numel() == param_bf16.numel()
    ), "param_fp32 and param_bf16 must have the same number of elements"
    assert (
        param_fp32.numel() == g_bf16.numel()
    ), "param_fp32 and g_fp16 must have the same number of elements"
    assert (
        param_fp32.numel() == m_fp32.numel()
    ), "param_fp32 and m_m_fp32 must have the same number of elements"
    assert (
        param_fp32.numel() == v_fp32.numel()
    ), "param_fp32 and v_fp32 must have the same number of elements"
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    stream = torch.cuda.current_stream().cuda_stream
    if not C.is_bf16_supported():
        raise NotImplementedError(f"bfloat16 is not supported on current GPU")
    C.adam_bf16_launcher(
        param_fp32.numel(),
        param_fp32.data_ptr(),
        param_bf16.data_ptr(),
        g_bf16.data_ptr(),
        m_fp32.data_ptr(),
        v_fp32.data_ptr(),
        beta1,
        beta2,
        eps,
        lr,
        scale,
        weight_decay,
        bias_correction1,
        bias_correction2,
        stream,
    )
