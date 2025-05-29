import os, glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

# change arch="80" to other code for your platform, see https://developer.nvidia.com/cuda-gpus#compute.
arch = "80"

setup(
    name='llamacu',
    version='0.0.0',
    author_email="acha131441373@gmail.com",
    description="llama cuda implementation",
    packages=find_packages(),
    setup_requires=[
        "pybind11",
        "psutil",
        "ninja",
    ],
    install_requires=[
        "transformers==4.46.2",
        "accelerate==0.26.0",
        "datasets",
        "fschat",
        "openai",
        "anthropic",
        "human_eval",
        "zstandard"
    ],
    ext_modules=[
        CUDAExtension(
            name='llamacu.C',
            sources = [
                "src/entry.cu",
                "src/utils.cu",
                # *glob.glob("src/flash_attn/src/*.cu"),
                *glob.glob("src/flash_attn/src/*hdim64_fp16*.cu"),
                *glob.glob("src/flash_attn/src/*hdim128_fp16*.cu"),
                *glob.glob("src/flash_attn/src/*hdim64_bf16*.cu"),
                *glob.glob("src/flash_attn/src/*hdim128_bf16*.cu"),
            ],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    [
                        "-O3", "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
                        f"-D_ARCH{arch}",
                    ]
                ),
            },
            include_dirs=[
                f"{this_dir}/src/flash_attn",
                f"{this_dir}/src/flash_attn/src",
                f"{this_dir}/src/cutlass/include",
            ],
        )
    ],
    cmdclass={
        'build_ext': NinjaBuildExtension
    }
) 