from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from config import settings

class DressClassifier:
    def __init__(self,model_name:str=settings.CLIP_MODEL_PATH,device: str = "cpu"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.labels=["formal","informal"]
        self.text_prompt = [
                "a photo of a person wearing formal corporate attire including a formal shirt, trousers, blazer, suit, and professional office wear and  tailcoats ",

               "a photo of a person wearing informal or casual attire including a t-shirt, jeans, hoodie, casual shirt, or other casual wear"
            ]

    def classify(self,crop:np.ndarray)->Dict[str,Any]:
        rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)   

        inputs = self.processor(
        text=self.text_prompt, 
        images=pil_image,
        return_tensors="pt",
        padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        formal_prob = float(probs[0].item())
        informal_prob = float(probs[1].item())

        if formal_prob >= informal_prob:
            predict_lable="formal"
            confidence=formal_prob
        else:
            predict_lable="informal"
            confidence=informal_prob
   

        return  {
        "lable":predict_lable,
        "confidence":round(confidence,4)
    }



