# 安装

## 安装方法

### 1. 用 pip 安装 (推荐)

```shell
$ pip install bmtrain
```

### 2. 从源代码安装

```shell
$ git clone https://github.com/OpenBMB/BMTrain.git
$ cd BMTrain
$ python3 setup.py install
```

## 编译选项

通过设置环境变量，你可以控制BMTrain的编译选项（默认会自动适配编译环境）：

### AVX指令集

* 强制使用AVX指令集: `BMT_AVX256=ON`
* 强制使用AVX512指令集: `BMT_AVX512=ON`

### CUDA计算兼容性

`TORCH_CUDA_ARCH_LIST=6.0 6.1 7.0 7.5 8.0+PTX`

## 推荐配置

* 网络：Infiniband 100Gbps / RoCE 100Gbps
* GPU：NVIDIA Tesla V100 / NVIDIA Tesla A100 / RTX 3090
* CPU：支持AVX512指令集的CPU，32核心以上
* RAM：256GB以上

## 常见问题

如果在编译过程中如下的报错信息，请尝试使用更新版本的gcc编译器。
```
error: invalid static_cast from type `const torch::OrderdDict<...>`
```

