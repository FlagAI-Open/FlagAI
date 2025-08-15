# Installation

## Install BMTrain

### 1. From PyPI (Recommend)

```shell
$ pip install bmtrain
```

### 2. From Source

```shell
$ git clone https://github.com/OpenBMB/BMTrain.git
$ cd BMTrain
$ python3 setup.py install
```

## Compilation Options

By setting environment variables, you can configure the compilation options of BMTrain (by default, the compilation environment will be automatically adapted).

### AVX Instructions

* Force the use of AVX instructions: `BMT_AVX256=ON`
* Force the use of AVX512 instructions: `BMT_AVX512=ON`

### CUDA Compute Capability

`TORCH_CUDA_ARCH_LIST=6.0 6.1 7.0 7.5 8.0+PTX`

## Recommended Configuration

* Network：Infiniband 100Gbps / RoCE 100Gbps
* GPU：NVIDIA Tesla V100 / NVIDIA Tesla A100 / RTX 3090
* CPU：CPU that supports AVX512 instructions, 32 cores or above
* RAM：256GB or above

## FAQ

If the following error message is reported during compilation, try using a newer version of the gcc compiler.
```
error: invalid static_cast from type `const torch::OrderdDict<...>`
```

