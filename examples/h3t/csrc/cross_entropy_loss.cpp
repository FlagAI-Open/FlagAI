#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void cross_entropy_forward_launcher(int32_t m, int32_t n, const torch::Tensor &input, const torch::Tensor &target, torch::Tensor &softmax, torch::Tensor &output, int32_t ignore_index);
void cross_entropy_backward_launcher(int32_t m, int32_t n, const torch::Tensor &grad_output, const torch::Tensor &target, const torch::Tensor &softmax, torch::Tensor &grad_input, int32_t ignore_index);
void cross_entropy_forward_inplace_launcher(int32_t m, int32_t n, torch::Tensor &x, const torch::Tensor &target, torch::Tensor &output, int32_t ignore_index);
void cross_entropy_backward_inplace_launcher(int32_t m, int32_t n, const torch::Tensor &grad_output, const torch::Tensor &target, torch::Tensor &x, int32_t ignore_index);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

void F_cross_entropy_forward(
    int32_t m, int32_t n,
    const torch::Tensor &input,     // (m, n)
    const torch::Tensor &target,    // (m)
    torch::Tensor &softmax,         // (m, n)
    torch::Tensor &output,          // (m)
    int32_t ignore_index
) {
    CHECK_INPUT(input);
    CHECK_INPUT(target);
    CHECK_INPUT(softmax);
    CHECK_INPUT(output);
    AT_ASSERTM(input.dtype() == torch::kHalf, "input must be a half tensor");
    AT_ASSERTM(target.dtype() == torch::kInt, "target must be a int tensor");
    AT_ASSERTM(softmax.dtype() == torch::kHalf, "softmax must be a half tensor");
    AT_ASSERTM(output.dtype() == torch::kFloat, "output must be a float tensor");
    AT_ASSERTM(input.numel() == softmax.numel(), "input and softmax must have the same number of elements");
    AT_ASSERTM(target.numel() == output.numel(), "target and output must have the same number of elements");

    cross_entropy_forward_launcher(m, n, input, target, softmax, output, ignore_index);
}

void F_cross_entropy_backward(
    int32_t m, int32_t n,
    const torch::Tensor &grad_output,   // (m)
    const torch::Tensor &target,        // (m) 
    const torch::Tensor &softmax,       // (m, n)
    torch::Tensor &grad_input,          // (m, n)
    int32_t ignore_index
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(target);
    CHECK_INPUT(softmax);
    CHECK_INPUT(grad_input);
    AT_ASSERTM(grad_output.dtype() == torch::kFloat, "grad_output must be a float tensor");
    AT_ASSERTM(target.dtype() == torch::kInt, "target must be a int tensor");
    AT_ASSERTM(softmax.dtype() == torch::kHalf, "softmax must be a half tensor");
    AT_ASSERTM(grad_input.dtype() == torch::kHalf, "grad_input must be a half tensor");
    AT_ASSERTM(grad_input.numel() == softmax.numel(), "grad_input and softmax must have the same number of elements");
    AT_ASSERTM(target.numel() == grad_output.numel(), "target and grad_output must have the same number of elements");

    cross_entropy_backward_launcher(m, n, grad_output, target, softmax, grad_input, ignore_index);
}

void F_cross_entropy_forward_inplace(
    int32_t m, int32_t n,
    torch::Tensor &x,               // (m, n)
    const torch::Tensor &target,    // (m)
    torch::Tensor &output,          // (m)
    int32_t ignore_index
) {
    CHECK_INPUT(x);
    CHECK_INPUT(target);
    CHECK_INPUT(output);
    AT_ASSERTM(x.dtype() == torch::kHalf, "x must be a half tensor");
    AT_ASSERTM(target.dtype() == torch::kInt, "target must be a int tensor");
    AT_ASSERTM(output.dtype() == torch::kFloat, "output must be a float tensor");
    AT_ASSERTM(target.numel() == output.numel(), "target and output must have the same number of elements");

    cross_entropy_forward_inplace_launcher(m, n, x, target, output, ignore_index);
}

void F_cross_entropy_backward_inplace(
    int32_t m, int32_t n,
    const torch::Tensor &grad_output,   // (m)
    const torch::Tensor &target,        // (m) 
    torch::Tensor &x,                   // (m, n)
    int32_t ignore_index
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(target);
    CHECK_INPUT(x);
    AT_ASSERTM(grad_output.dtype() == torch::kFloat, "grad_output must be a float tensor");
    AT_ASSERTM(target.dtype() == torch::kInt, "target must be a int tensor");
    AT_ASSERTM(x.dtype() == torch::kHalf, "x must be a half tensor");
    AT_ASSERTM(target.numel() == grad_output.numel(), "target and grad_output must have the same number of elements");

    cross_entropy_backward_inplace_launcher(m, n, grad_output, target, x, ignore_index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_cross_entropy_forward", &F_cross_entropy_forward, "cross entropy forward");
    m.def("f_cross_entropy_backward", &F_cross_entropy_backward, "cross entropy backward");
    m.def("f_cross_entropy_forward_inplace", &F_cross_entropy_forward_inplace, "cross entropy forward inplace");
    m.def("f_cross_entropy_backward_inplace", &F_cross_entropy_backward_inplace, "cross entropy backward inplace");
}
