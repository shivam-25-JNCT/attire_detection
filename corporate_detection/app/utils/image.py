import os
import uuid
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

def decode_image_bytes(image_bytes:bytes) -> np.ndarray:
    np_arr=np.frombuffer(image_bytes,np.uint8)
    frame=cv2.imdecode(np_arr,cv2.IMREAD_COLOR)

    if frame is None:
       raise ValueError("Failed to decode image. Invalid image format.")
    return frame

def crop_persone (frame :np.ndarray,bbox:list[int] | Tuple[int,int,int,int])->np.ndarray:
    h,w,_=frame.shape
    x1,y1,x2,y2=bbox

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    crop=frame[y1:y2,x1:x2]
    if crop.size==0:
        raise ValueError(f"Invalid crop dimension genrated from bbox {bbox}")
    return crop

def save_crop_snapshot(crop:np.ndarray,base_dir:Path,persone_idx:int)->str:
    base_dir.mkdir(parants=True,exist_ok=True)
    filename=f"persone_{persone_idx}_{uuid.uuid4().hex[:8]}.jpg"
    filepath=base_dir / filename
    return str(filename)