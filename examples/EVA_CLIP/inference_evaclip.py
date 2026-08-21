import torch
import urllib
import io
from PIL import Image
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.dataset.mm.clip_dataset import clip_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = AutoLoader(task_name="txt_img_matching", #contrastive learning
                    model_name="eva-clip")

model = loader.get_model()
model.eval()
model.to(device)
tokenizer = loader.get_tokenizer()
transform = clip_transform(img_size=model.visual.image_size)

def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream

def inference():
    # local image
    # image = Image.open(/path/to/image)
    # online image
    image = Image.open(download_image("https://bkimg.cdn.bcebos.com/pic/4610b912c8fcc3ce2d02315d9d45d688d53f209a?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxMTY=,g_7,xp_5,yp_5"))
    image = transform(image).unsqueeze(0).to(device)
    text = tokenizer.tokenize_as_tensor(["a tomato", "a cat"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)

    print(text_probs.cpu().numpy()[0].tolist()) # [1.0, 0.0]

if __name__=="__main__":
    inference()
