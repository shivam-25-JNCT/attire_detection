from typing import Optional,Generator
import cv2
import numpy as np

class RTSPStreamReader:
    def __init__(self,rtsp_url:str):
        self.rtsp_url=rtsp_url
        self.cap=Optional[cv2.VideoCapture]=None

    def connect(self)->bool :
        self.cap=cv2.VideoCapture(self.rtsp_url)
        return self.cap.isOpened()
    
    def read_frame(self)->Optional[np.ndarray]:
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                return None
        rat,frame=self.cap.read()
        if not rat:
            return None
        return frame

    def frame_genrator(self)-> Generator(np.ndarray,None,None):
        if not self.connect():
            raise RuntimeError(f"Could not open RTSP stream at: {self.rtsp_url}")
        try:
            while True:
                ret,frame=self.cap.read()
                if not ret:
                    break
                yield frame
        finally :
            self.release()

    def release (self):
        if self.cap and self.cap.isOpened():
           self.cap.release()