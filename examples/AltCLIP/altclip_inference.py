import torch
from PIL import Image
from flagai.auto_model.auto_loader import AutoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = AutoLoader(
    task_name="txt_img_matching",
    model_name="AltCLIP-XLMR-L",   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
    model_dir="./checkpoints"
)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
transform = loader.get_transform()

model.eval()
model.to(device)
tokenizer = loader.get_tokenizer()

def inference():
    image = Image.open("./dog.jpeg")
    image = transform(image)
    image = torch.tensor(image["pixel_values"]).to(device)
    tokenizer_out = tokenizer(["a rat", "a dog", "a cat"], 
                                padding=True,
                                truncation=True,
                                max_length=77,
                                return_tensors='pt')

    text = tokenizer_out["input_ids"].to(device)
    attention_mask = tokenizer_out["attention_mask"].to(device)
    with torch.no_grad():
        image_features = model.get_image_features(image)
        text_features = model.get_text_features(text, attention_mask=attention_mask)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)

    print(text_probs.cpu().numpy()[0].tolist())

if __name__=="__main__":
    inference()
