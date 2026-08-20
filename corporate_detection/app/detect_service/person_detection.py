from ultralytics import YOLO
from typing import List,Dict,Any
import numpy as np
import torch


class PersoneDetector:
    def __init__(self,model_path:str,conf_thresh:float=0.50,device:str="cpu"):
        self.device = device if torch.cuda.is_available() and device=="cuda" else "cpu"
        self.model=YOLO(model_path)
        self.conf_thresh=conf_thresh

    def detect (self, frame:np.ndarray)->List[Dict[str,any]]:
        results = self.model.track(
    source=frame,
    classes=[0],
    conf=self.conf_thresh,
    persist=True,
    tracker="bytetrack.yaml",
    device=self.device,
    verbose=False
)   
        detections=[]
        if not results and results[0].boxes is None:
            return detections

        first_res=results[0]
        boxes=first_res.boxes
        for box in boxes:
            if box.id is not None:
                track_id=int(box.id[0].cpu().numpy())
            else:
                track_id = -1
            xyxy=box.xyxy[0].cpu().numpy().astype(int).tolist()
            confidence=float(box.conf[0].cpu().numpy())

            detections.append({
                "track_id": track_id,
                "bbox": xyxy,  # [x1, y1, x2, y2]
                "confidence": round(confidence, 4)
            })
        return detections
    




