
#  模型信息 Model Information

我们使用预训练好的双语语言模型作为我们的text encoder，并使用美学评分在5.5分以上的WuDao数据集(6M)和美学评分在以上的Laion数据(5M)进行微调。

微调时，我们使用stable-diffusion-v1-4作初始化，并冻住双语语言模型，只微调Unet模型中Transformer Block的key模块与vuale模块。

并且在训练时，我们将数据集按图片的长宽比进行分桶，同一个batch内的数据都裁剪到与图片尺寸相近的固定大小，从而克服了原版stable diffusion长图与宽图生成多头的问题。

我们的版本在中英文对齐方面表现非常出色，是目前市面上开源的最强版本，保留了原版stable diffusion的大部分能力，并且在某些例子上比有着比原版模型更出色的能力。

AltDiffusion 模型由名为 AltCLIP 的双语 CLIP 模型支持，该模型也可在本项目中访问。您可以阅读 [此教程](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) 了解更多信息。

注意：模型推理要求一张至少10G以上的GPU。

We use the pre-trained bilingual language model as our text encoder and fine-tune it using the WuDao dataset (6M) with an aesthetic score above 5.5 and the Laion data (5M) with an aesthetic score above 5.5.

When fine-tuning, we use stable-diffusion v1-4 as initialization, freeze the bilingual language model, and only fine-tune the key module and vuale module of the Transformer Block in the Unet model.

And during training, we divide the data set into bucketed according to the aspect ratio of the image, and the data in the same batch are cropped to a fixed size similar to the image size, so as to overcome the problem of generating multiple heads for the long and wide images of the original stable diffusion.

AltDiffusion model is backed by a bilingual CLIP model named AltCLIP, which is also accessible in FlagAI. You can read [this tutorial](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) for more information. 

Note that the model inference requires a GPU of at least 10G above.



# 示例 Example

以下示例将为文本输入`Anime portrait of natalie portman as an anime girl by stanley artgerm lau, wlop, rossdraws, james jean, andrei riabovitchev, marc simonetti, and sakimichan, trending on artstation` 在目录`./AltDiffusionOutputs`下生成图片结果。

The following example will generate image results for text input `Anime portrait of natalie portman as an anime girl by stanley artgerm lau, wlop, rossdraws, james jean, andrei riabovitchev, marc simonetti, and sakimichan, trending on artstation` under the default output directory `./AltDiffusionOutputs`

```python
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

# Initialize 
prompt = "Anime portrait of natalie portman as an anime girl by stanley artgerm lau, wlop, rossdraws, james jean, andrei riabovitchev, marc simonetti, and sakimichan, trending on artstation"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader = AutoLoader(task_name="text2img", #contrastive learning
                    model_name="AltDiffusion",
                    model_dir="./checkpoints")

model = loader.get_model()
model.eval()
model.to(device)
predictor = Predictor(model)
predictor.predict_generate_images(prompt)
```

您可以在`predict_generate_images`函数里通过改变参数来调整设置，具体信息如下:

More parameters of predict_generate_images for you to adjust for `predict_generate_images` are listed below:


`prompt: str`: 提示文本; The prompt text

`out_path: str`: 输出路径; The output path to save images

`n_samples: int`: 输出图片数量; Number of images to be generated

`skip_grid: bool`: 如果为True, 会将所有图片拼接在一起，输出一张新的图片; If set to true, image gridding step will be skipped

`ddim_step: int`: DDIM模型的步数; Number of steps in ddim model

`plms: bool`: 如果为True, 则会使用plms模型; If set to true, PLMS Sampler instead of DDIM Sampler will be applied

`scale: float` : 这个值决定了文本在多大程度上影响生成的图片，值越大影响力越强; This value determines how important the prompt incluences generate images

`H: int`: 图片的高度; Height of image

`W: int`: 图片的宽度; Width of image

`C: int`: 图片的channel数; Numeber of channels of generated images

`seed: int`: 随机种子; Random seed number 

# 模型权重 Model Weights

第一次运行AltDiffusion模型时会自动下载下列权重:

The following weights are automatically downloaded when the AltDiffusion model is run for the first time: 

| 模型名称 Model name              | 大小 Size | 描述 Description                                        |
|------------------------------|---------|-------------------------------------------------------|
| StableDiffusionSafetyChecker | 1.13G   | 图片的安全检查器；Safety checker for image                     |
| AltDiffusion                 | 8.0G    | 我们的双语AltDiffusion模型； Our bilingual AltDiffusion model |
| AltCLIP                      | 3.22G   | 我们的双语AltCLIP模型；Our bilingual AltCLIP model            |

# 更多生成结果 More Results

## 中英文对齐能力

### prompt:dark elf princess, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and fuji choko and viktoria gavrilenko and hoang lap
### 英文生成结果

![image](./imgs/en_暗黑精灵.png)

### prompt:黑暗精灵公主，非常详细，幻想，非常详细，数字绘画，概念艺术，敏锐的焦点，插图
### 中文生成结果
![image](./imgs/cn_暗黑精灵.png)

## 中文表现能力

## prompt:带墨镜的男孩肖像，充满细节，8K高清
![image](./imgs/小男孩.png)


## prompt:带墨镜的中国男孩肖像，充满细节，8K高清
![image](./imgs/cn_小男孩.png)

## 长图生成能力

### prompt: 一只带着帽子的小狗 
### 原版 stable diffusion：
![image](./imgs/多尺度狗（不好）.png)

### Ours:
![image](./imgs/多尺度狗（好）.png)

注: 此处长图生成技术由右脑科技(RightBrain AI)提供。

Note: Note: The long image generation technology here is provided by Right Brain Technology.

# License

The model is licensed with a [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license). The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. You can modify and use the model for commercial purposes, but a copy of the same use restrictions must be included. For the full list of restrictions please [read the license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)