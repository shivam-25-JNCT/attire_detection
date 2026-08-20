from ultralytics import YOLO
from typing import Dict,Any
import numpy as np
import torch


class IDCardDetector:
    def __init__(self,model_path:str,conf_thresh:float=0.45,device:str="cpu"):
        self.device = device if torch.cuda.is_available() and device=="cuda" else "cpu"
        self.model=YOLO(model_path)
        self.conf_thresh=conf_thresh

    def detect (self, crop:np.ndarray)->Dict[str,any]:
        results = self.model.predict(
    source=crop,
   
    conf=self.conf_thresh,
    device=self.device,
    verbose=False
)   
        
        if not results or len(results[0].boxes)==0:
            return {
                "detected":False,
                "confidence":0.0
            }

        
        max_conf=float(results[0].boxes.conf.max().cpu().numpy())
        
        return {
            "detected": True,
            "confidence": round(max_conf, 4)
        }
    




