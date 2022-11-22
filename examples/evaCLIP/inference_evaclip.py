import torch
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

def inference():
    image = Image.open("./CLIP.png")
    image = transform(image).unsqueeze(0).to(device)
    text = tokenizer.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)

    print(text_probs.cpu().numpy()[0].tolist())

if __name__=="__main__":
    inference()
