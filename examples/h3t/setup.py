from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

def get_avx_flags():
    if os.environ.get("BMT_AVX256", "").lower() in ["1", "true", "on"]:
        return ["-mavx", "-mfma", "-mf16c"]
    elif os.environ.get("BMT_AVX512", "").lower() in ["1", "true", "on"]:
        return ["-mavx512f"]
    else:
        return ["-march=native"]

def get_device_cc():
    try:
        CC_SET = set()
        for i in range(torch.cuda.device_count()):
            CC_SET.add(torch.cuda.get_device_capability(i))
        
        if len(CC_SET) == 0:
            return None
        
        ret = ""
        for it in CC_SET:
            if len(ret) > 0:
                ret = ret + " "
            ret = ret + ("%d.%d" % it)
        return ret
    except RuntimeError:
        return None

avx_flag = get_avx_flags()
device_cc = get_device_cc()
if device_cc is None:
    if not torch.cuda.is_available():
        os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5 8.0+PTX")
    else:
        if torch.version.cuda.startswith("10"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5+PTX")
        else:
            if not torch.version.cuda.startswith("11.0"):
                os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5 8.0 8.6+PTX")
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "6.0 6.1 7.0 7.5 8.0+PTX")
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", device_cc)

ext_modules = []

if os.environ.get("GITHUB_ACTIONS", "false") == "false":
    ext_modules = [
        CUDAExtension('bmtrain.nccl._C', [
            'csrc/nccl.cpp',
        ], include_dirs=["csrc/nccl/build/include"], extra_compile_args={}),
        CUDAExtension('bmtrain.optim._cuda', [
            'csrc/adam_cuda.cpp',
            'csrc/cuda/adam.cu',
            'csrc/cuda/has_inf_nan.cu'
        ], extra_compile_args={}),
        CppExtension("bmtrain.optim._cpu", [
            "csrc/adam_cpu.cpp",
        ], extra_compile_args=[
            '-fopenmp', 
            *avx_flag
        ], extra_link_args=['-lgomp']),
        CUDAExtension('bmtrain.loss._cuda', [
            'csrc/cross_entropy_loss.cpp',
            'csrc/cuda/cross_entropy.cu',
        ], extra_compile_args={}),
    ]
else:
    ext_modules = []

setup(
    name='bmtrain',
    version='0.1.8',
    author="Guoyang Zeng",
    author_email="qbjooo@qq.com",
    description="A toolkit for training big models",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    })

