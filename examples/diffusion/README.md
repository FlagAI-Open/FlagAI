Stable diffusion is a powerful text-to-image model, which can generate high-quality images given any text input. Running the following code will automatically download the model and produce 4 images under /open_CNCLIP_samples. The settings file is `config.yaml` under the download path, and can be modified according to your needs. 

Note that our stable diffusion model has a size of 11.4 G, and requires A100 GPU with at least 32G memory to run.

The following example will generate image results for text input `两只老虎` under `./open_CNCLIP_samples`

```python
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor

# Initialize 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="text2img", #contrastive learning
                    model_name="diffusion-ddpm-cnclip",)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
model.eval()
model.to(device)
predictor = Predictor(model, tokenizer)
predictor.predict_generate_images("两只老虎")
```

More parameters of predict_generate_images for you to adjust:


`prompt`: the prompt text

`out_path`: the output path to save images

`n_samples`: number of images to be generated

`skip_grid`: not to grid images

`ddim_step`: number of steps in ddim model

`H`: height of image

`W`: width of image

`C`: channels of images
