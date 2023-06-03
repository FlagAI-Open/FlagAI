# AltCLIP 的 ONNX 模型导出

## ONNX 是什么？

ONNX(Open Neural Network Exchange)，开放神经网络交换，用于在各种深度学习训练和推理框架转换的一个中间表示格式。

在实际业务中，可以使用 Pytorch 或者 TensorFlow 训练模型，导出成 ONNX 格式，然后用 ONNX Runtime 直接运行 ONNX。

使用 ONNX 可以减少模型的依赖，降低部署成本。

也可以进一步借助 ONNX 转换成目标设备上运行时支撑的模型格式，比如 [TensorRT Engine](https://developer.nvidia.com/tensorrt)、[NCNN](https://github.com/Tencent/ncnn)、[MNN](https://github.com/alibaba/MNN) 等格式， 优化性能。

## 下载 AltCLIP 的 ONNX

可以从[xxai/AltCLIP](https://huggingface.co/xxai/AltCLIP/tree/main)下载打包好的 onnx，并解压到 `FlagAI/onnx/onnx/` 下。

如此就可以直接运行 onnx 的测试，而无需下载原始模型运行导出。

## 文件说明

因为 flagai 的依赖复杂，所以构建容器便于导出。

### 脚本

* `./build.sh` 在本地构建容器

    可设置环境变量 `ORG=xxai` 使用 [hub.docker.com 上的已构建的镜像](https://hub.docker.com/repository/docker/xxai/altclip-onnx)。

    比如，运行 `ORG=xxai ./bash.sh` 。

* `./bash.sh` 在本地进入容器的 bash，方便调试

* `./export.sh` 运行容器，下载 pytorch 模型，然后转换为 onnx

    设置环境变量 MODEL ，可以配置导出、测试脚本运行的模型 。

    默认导出的模型是 AltCLIP-XLMR-L-m18 ，还支持以下模型：

    * AltCLIP-XLMR-L
    * AltCLIP-XLMR-L-m9

    运行后将生成 4 个 onnx 文件和很多权重文件

    * onnx/AltCLIP-XLMR-L-m18/onnx/Img.onnx
    * onnx/AltCLIP-XLMR-L-m18/onnx/ImgNorm.onnx
    * onnx/AltCLIP-XLMR-L-m18/onnx/Txt.onnx
    * onnx/AltCLIP-XLMR-L-m18/onnx/TxtNorm.onnx

    其中 Norm 代表输出归一化的向量，如果想把生成的文本向量和图片向量存入向量数据库，进行相似性搜索，请用归一化的向量。

    具体用见下文的 onnx 模型的测试脚本。

* `./dist.sh` 运行容器，导出以上 3 个模型的 onnx，并打包放到 dist 目录下。

### 目录

* `model/` 存放下载的模型
* `onnx/` 存放导出的 onnx，下载的 onnx 也请解压到这里

### 测试

#### onnx 模型的依赖安装

test/onnx 下面的依赖很简单，只有 transformers 和 onnxruntime，不依赖于 flagai。

onnxruntime 有很多版本可以选择，见[onnxruntime](https://onnxruntime.ai/)。

对于 python 而言，常见的运行时推荐如下：

* 显卡 `pip install onnxruntime-gpu`
* ARM 架构的 MAC `pip install onnxruntime-silicon` (目前还不支持 python3.11)
* INTEL 的 CPU `pip install onnxruntime-openvino`
* 其他 CPU `pip install onnxruntime`

运行 [./test/onnx/setup.sh](./test/onnx/setup.sh) 会自动判断环境，选择安装合适的 onnxruntime 版本和 transformers。

#### onnx 模型的测试脚本

请先安装 [direnv](https://github.com/direnv/direnv/blob/master/README.md) 并在本目录下 `direnv allow` 或者手工 `source .envrc` 来设置 PYTHONPATH 环境变量。

* [./test/onnx/onnx_img.py](./test/onnx/onnx_img.py)  生成图片向量 (norm 代表归一化的向量，可用于向量搜索)
* [./test/onnx/onnx_txt.py](./test/onnx/onnx_txt.py)  生成文本向量
* [./test/onnx/onnx_test.py](./test/onnx/onnx_test.py)

  匹配图片向量和文本向量，进行零样本分类

  可借助向量数据库，提升零样本分类的准确性，参见[ECCV 2022 | 无需下游训练，Tip-Adapter 大幅提升 CLIP 图像分类准确率](https://cloud.tencent.com/developer/article/2126102)。
* [./test/onnx/onnx_load.py](./test/onnx/onnx_load.py)

  onnx 模型的加载代码，运行它可以看到当前机器可用的 onnx provider。

  比如苹果 M2 芯片的笔记本上运行如下：

  ```
  ❯ ./onnx_load.py 2>/dev/null
  all providers :
  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'MIGraphXExecutionProvider', 'ROCMExecutionProvider', 'OpenVINOExecutionProvider', 'DnnlExecutionProvider', 'TvmExecutionProvider', 'VitisAIExecutionProvider', 'NnapiExecutionProvider', 'CoreMLExecutionProvider', 'ArmNNExecutionProvider', 'ACLExecutionProvider', 'DmlExecutionProvider', 'RknpuExecutionProvider', 'XnnpackExecutionProvider', 'CANNExecutionProvider', 'CPUExecutionProvider']

  now can use providers :
  ['CoreMLExecutionProvider', 'CPUExecutionProvider']
  ```

  可以创建 FlagAI/onnx/.env ，设置环境变量 `ONNX_PROVIDER`，配置当前环境的 Onnx Execution Provider，方便测试对比性能。

  设置的示例如下：

  ```
  ❯ cat FlagAI/onnx/.env
  ONNX_PROVIDER=CoreMLExecutionProvider
  ```

  设置成功后，需要在 `FlagAI/onnx` 目录下运行 `direnv allow` 或者手工 `source .envrc` 让其在当前命令行中生效。

#### pytorch 模型

用于对比 onnx 模型的向量输出，查看是否一致。

因为用到了 flagai，请如下图所示运行 [./bash.sh ](./bash.sh) 进入容器运行调试。

![](https://pub-b8db533c86124200a9d799bf3ba88099.r2.dev/2023/06/ei64CNo.webp)

* [./test/clip/clip_img.py](./test/clip/clip_img.py)  生成图片向量
* [./test/clip/clip_txt.py](./test/clip/clip_txt.py)  生成文本向量
* [./test/clip/clip_test.py](./test/clip/clip_test.py) 匹配图片向量和文本向量，进行零样本分类
