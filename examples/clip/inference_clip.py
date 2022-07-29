import sys
sys.path.append("/mnt/wchh/FlagAI-internal")
import torch
from PIL import Image
from flagai.model.clip_model import CLIP
from flagai.data.transform import image_transform #文件位置待确定
from flagai.data.tokenizer.clip import tokenizer
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dir = "/root/.cache/clip"#'/mnt/clip_models/ViT-B-32'
dir = "/mnt/clip_models/ViT-B-32"

def test_inference():
    model = CLIP.init_from_json(os.path.join(dir,"config.json")).to(device)
    preprocess = image_transform(model.visual.image_size, is_train=False)

    model_path = os.path.join(dir, "pytorch_model.bin")

    model.load_state_dict(torch.load(model_path), strict=False)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "CLIP.png")).unsqueeze(0).to(device)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T)
    print(text_probs)
    assert text_probs.cpu().numpy()[0].tolist() == [1.0, 0.0, 0.0]

if __name__=="__main__":
    test_inference()