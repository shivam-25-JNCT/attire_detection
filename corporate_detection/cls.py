from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Model load
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(r"F:\person_detection\service\img\casual.png").convert("RGB")

# Better prompts (important)
texts = [
    "a person wearing formal clothes, suit, blazer, formal shirt and trousers",
    "a person wearing casual clothes, t-shirt, jeans, hoodie, informal wear"
]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

formal_score = probs[0].item()
casual_score = probs[1].item()

print(f"Formal  : {formal_score:.2%}")
print(f"Casual  : {casual_score:.2%}")

if formal_score > casual_score:
    print("→ Final Prediction: Formal")
else:
    print("→ Final Prediction: Casual")