import os, glob
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))

def detect_cuda_arch():
    """Automatically detect current CUDA architecture"""
    # 1. First check if environment variable specifies architecture
    env_arch = os.getenv("CPMCU_CUDA_ARCH")
    if env_arch:
        # Only support simple comma-separated format, e.g., "80,86"
        arch_list = []
        tokens = env_arch.split(',')
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
                
            # Check format: must be pure digits
            if not token.isdigit():
                raise ValueError(
                    f"Invalid CUDA architecture format: '{token}'. "
                    f"CPMCU_CUDA_ARCH should only contain comma-separated numbers like '80,86'"
                )
            
            arch_list.append(token)
        
        if arch_list:
            print(f"Using CUDA architectures from environment variable: {arch_list}")
            return arch_list
    
    # 2. Check if torch library is available, if so, auto-detect
    try:
        import torch
    except ImportError:
        # 3. If no environment variable and no torch, throw error
        raise RuntimeError(
            "CUDA architecture detection failed. Please either:\n"
            "1. Set environment variable CPMCU_CUDA_ARCH (e.g., export CPMCU_CUDA_ARCH=90), or\n"
            "2. Install PyTorch (pip install torch) for automatic detection.\n"
            "Common CUDA architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100)"
        )
    
    # Use torch to auto-detect all GPU architectures
    try:
        if torch.cuda.is_available():
            arch_set = set()
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                major, minor = torch.cuda.get_device_capability(i)
                arch = f"{major}{minor}"
                arch_set.add(arch)
            
            arch_list = sorted(list(arch_set))  # Sort for consistency
            print(f"Detected CUDA architectures: {arch_list} (from {device_count} GPU devices)")
            return arch_list
        else:
            raise RuntimeError(
                "No CUDA devices detected. Please either:\n"
                "1. Set environment variable CPMCU_CUDA_ARCH (e.g., export CPMCU_CUDA_ARCH=90), or\n"
                "2. Ensure CUDA devices are available and properly configured.\n"
                "Common CUDA architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100)"
            )
    except Exception as e:
        raise RuntimeError(
            f"CUDA architecture detection failed: {e}\n"
            "Please set environment variable CPMCU_CUDA_ARCH (e.g., export CPMCU_CUDA_ARCH=90).\n"
            "Common CUDA architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100)"
        )

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "16"
    return nvcc_extra_args + ["--threads", nvcc_threads]

def get_compile_args():
    """Return different compilation arguments based on debug mode"""
    debug_mode = os.getenv("CPMCU_DEBUG", "0").lower() in ("1", "true", "yes")
    perf_mode = os.getenv("CPMCU_PERF", "0").lower() in ("1", "true", "yes")
    
    # Check precision type environment variable
    dtype_env = os.getenv("CPMCU_DTYPE", "fp16").lower()
    
    # Parse precision type list
    dtype_list = [dtype.strip() for dtype in dtype_env.split(',')]
    
    # Validate precision types
    valid_dtypes = {"fp16", "bf16"}
    invalid_dtypes = [dtype for dtype in dtype_list if dtype not in valid_dtypes]
    if invalid_dtypes:
        raise ValueError(
            f"Invalid CPMCU_DTYPE values: {invalid_dtypes}. "
            f"Supported values: 'fp16', 'bf16', 'fp16,bf16'"
        )
    
    # Deduplicate and generate compilation definitions
    dtype_set = set(dtype_list)
    dtype_defines = []
    if "fp16" in dtype_set:
        dtype_defines.append("-DENABLE_DTYPE_FP16")
    if "bf16" in dtype_set:
        dtype_defines.append("-DENABLE_DTYPE_BF16")
    
    # Print compilation information
    if len(dtype_set) == 1:
        dtype_name = list(dtype_set)[0].upper()
        print(f"Compiling with {dtype_name} support only")
    else:
        dtype_names = [dtype.upper() for dtype in sorted(dtype_set)]
        print(f"Compiling with {' and '.join(dtype_names)} support")
    
    # Common compilation arguments
    common_cxx_args = ["-std=c++17"] + dtype_defines
    common_nvcc_args = [
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ] + dtype_defines
    
    if debug_mode:
        print("Debug mode enabled (CPMCU_DEBUG=1)")
        cxx_args = common_cxx_args + [
            "-g3",           # Most detailed debug information
            "-O0",           # Disable optimization
            "-DDISABLE_MEMPOOL",
            "-DDEBUG", 
            "-fno-inline",   # Disable inlining
            "-fno-omit-frame-pointer",  # Keep frame pointer
        ]
        nvcc_base_args = common_nvcc_args + [
            "-O0", 
            "-g",            # Host-side debug information
            "-lineinfo",     # Generate line number information
            "-DDISABLE_MEMPOOL",
            "-DDEBUG", 
            "-DCUDA_DEBUG",
            "-Xcompiler", "-g3",              # Pass to host compiler
            "-Xcompiler", "-fno-inline",      # Disable inlining
            "-Xcompiler", "-fno-omit-frame-pointer",  # Keep frame pointer
        ]
        # Debug mode linking arguments
        link_args = ["-g", "-rdynamic"]
    else:
        print("Release mode enabled")
        cxx_args = common_cxx_args + ["-O3"]
        nvcc_base_args = common_nvcc_args + [
            "-O3",
            "--use_fast_math",
        ]
        # Release mode linking arguments
        link_args = []
    
    # Add performance testing control
    if perf_mode:
        print("Performance monitoring enabled (CPMCU_PERF=1)")
        cxx_args.append("-DENABLE_PERF")
        nvcc_base_args.append("-DENABLE_PERF")
    else:
        print("Performance monitoring disabled (CPMCU_PERF=0)")
    
    return cxx_args, nvcc_base_args, link_args, dtype_set

def get_all_headers():
    """Get all header files for dependency tracking"""
    header_patterns = [
        "src/**/*.h",
        "src/**/*.hpp", 
        "src/**/*.cuh",
        "src/cutlass/include/**/*.h",
        "src/cutlass/include/**/*.hpp",
        "src/flash_attn/**/*.h",
        "src/flash_attn/**/*.hpp",
        "src/flash_attn/**/*.cuh",
    ]
    
    headers = []
    for pattern in header_patterns:
        abs_headers = glob.glob(os.path.join(this_dir, pattern), recursive=True)
        # Convert to relative paths
        rel_headers = [os.path.relpath(h, this_dir) for h in abs_headers]
        headers.extend(rel_headers)
    
    # Filter out non-existent files (check absolute path but return relative path)
    headers = [h for h in headers if os.path.exists(os.path.join(this_dir, h))]
    
    return headers

def get_flash_attn_sources(enabled_dtypes):
    """Get flash attention source files based on enabled data types"""
    sources = []
    
    for dtype in enabled_dtypes:
        if dtype == "fp16":
            # sources.extend(glob.glob("src/flash_attn/src/*hdim64_fp16*.cu"))
            sources.extend(glob.glob("src/flash_attn/src/*hdim128_fp16*.cu"))
        elif dtype == "bf16":
            # sources.extend(glob.glob("src/flash_attn/src/*hdim64_bf16*.cu"))
            sources.extend(glob.glob("src/flash_attn/src/*hdim128_bf16*.cu"))
    
    return sources

# Try to build extension modules
ext_modules = []
cmdclass = {}

try:
    # Try to import torch-related modules
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    # Get CUDA architecture
    arch_list = detect_cuda_arch()
    
    # Get compilation arguments
    cxx_args, nvcc_base_args, link_args, dtype_set = get_compile_args()
    
    # Get header files
    all_headers = get_all_headers()
    
    # Generate gencode arguments for each architecture
    gencode_args = []
    arch_defines = []
    for arch in arch_list:
        gencode_args.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
        arch_defines.append(f"-D_ARCH{arch}")
    
    print(f"Using CUDA architecture compile flags: {arch_list}")
    
    flash_attn_sources = get_flash_attn_sources(dtype_set)
    
    # Create Ninja build extension class
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
    
    # Configure extension module
    ext_modules = [
        CUDAExtension(
            name='cpmcu.C',
            sources = [
                "src/entry.cu",
                "src/utils.cu",
                "src/signal_handler.cu",
                "src/perf.cu",
                *glob.glob("src/qgemm/gptq_marlin/*cu"),
                *flash_attn_sources,
            ],
            libraries=["cublas", "dl"],
            depends=all_headers,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": append_nvcc_threads(
                    nvcc_base_args + 
                    gencode_args +
                    arch_defines + [
                        # Add dependency file generation options
                        "-MMD", "-MP",
                    ]
                ),
            },
            extra_link_args=link_args,
            include_dirs=[
                f"{this_dir}/src/flash_attn",
                f"{this_dir}/src/flash_attn/src",
                f"{this_dir}/src/cutlass/include",
                f"{this_dir}/src/",
            ],
        )
    ]
    
    cmdclass = {'build_ext': NinjaBuildExtension}
    
except Exception as e:
    print(f"Warning: Unable to configure CUDA extension module: {e}")
    print("Skipping extension module build, installing Python package only...")

setup(
    name='cpmcu',
    version='1.0.0',
    author_email="acha131441373@gmail.com",
    description="cpm cuda implementation",
    packages=find_packages(),
    setup_requires=[
        "pybind11",
        "psutil",
        "ninja",
        "torch",
    ],
    install_requires=[
        "transformers==4.46.2",
        "accelerate==0.26.0",
        "datasets",
        "fschat",
        "openai",
        "anthropic",
        "human_eval",
        "zstandard",
        "tree_sitter",
        "tree-sitter-python"
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
) 