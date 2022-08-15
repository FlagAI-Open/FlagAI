import torch
import sys
sys.path.append("/mnt/wchh/FlagAI-internal")
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from flagai.auto_model.auto_loader import AutoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(task_name="cl", #contrastive learning
                    model_name="clip-base-p32-224",
                    model_dir="/mnt/clip_models/")

model = loader.get_model()
model.eval()
model.to(device)
tokenizer = loader.get_tokenizer()
n_px = model.image_size
print(n_px)
def _convert_image_to_rgb(image):
    return image.convert("RGB")
image_transform = Compose([
    Resize(n_px, interpolation=Image.BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def test_inference():
    image = Image.open("./CLIP.png")
    image = image_transform(image).unsqueeze(0).to(device)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    import  numpy as np
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)

    print(text_probs.cpu().numpy()[0].tolist())

if __name__=="__main__":
    test_inference()