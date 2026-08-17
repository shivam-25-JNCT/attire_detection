from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

class DressClassifier:
    def __init__(self,model_name:str="openai/clip-vit-base-patch32",device: str = "cpu"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
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