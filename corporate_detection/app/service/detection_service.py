from datetime import datetime , timezone
from typing import Dict,Any,Tuple,Optional
import numpy as np
from config import settings
from detect_service.person_detection import PersoneDetector
from detect_service.id_card_detection import IDCardDetector
from classification.dress_detect import DressClassifier
from utils.image import crop_persone,save_crop_snapshot
class DetectionService:
    def __init__(self,
        person_detector=PersoneDetector,
        id_card_detector=IDCardDetector,
        dress_claassifier=DressClassifier,
        roi_zone: Optional[Tuple[int, int, int, int]] = None
        ):
        self.person_detector=person_detector
        self.id_card_detector=id_card_detector
        self.dress_claassifier=dress_claassifier
        self.roi_zone = roi_zone
        self.processed_persons: Dict[int, Dict[str, Any]] = {}

    def set_roi(self, x1: int, y1: int, x2: int, y2: int):
        
        self.roi_zone = (x1, y1, x2, y2)

    def _is_in_roi(self, bbox: list) -> bool:
        if not self.roi_zone:
            return True  
        x1, y1, x2, y2 = bbox

        
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        rx1, ry1, rx2, ry2 = self.roi_zone
        # crx=int((rx1+rx2)/2)
        # cry=int((ry1+ry2)/2)
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        # return cx== crx and cy==cry
    
    def process_frame(self,frame:np.ndarray,camera_id:str="CAM-01",roi=None) ->Dict[str,any]:
        if roi:
            rx1, ry1, rx2, ry2 = roi
            
        

        person_detections=self.person_detector.detect(frame)
        detected_persone_payload=[]
        new_event = []
        
        for idx,data in enumerate(person_detections,start=1):
            bbox=data["bbox"]
            track_id=data["track_id"]
            
            if track_id == -1:
                continue
            in_roi = self._is_in_roi(bbox)

            if in_roi and (track_id not in self.processed_persons):

                try:
                    crop=crop_persone(frame,bbox)
                

                    dress_result=self.dress_claassifier.classify(crop)
                    id_card_result=self.id_card_detector.detect(crop)

                    snapshot_path=None 

                    if settings.SAVE_SNAPSHOT:
                        snapshot_path=save_crop_snapshot(crop,settings.SNAPSHOT_DIR,idx)

                    person_data = {
                        "dress": dress_result,
                        "id_card": id_card_result,
                        "snapshot": snapshot_path,
                        "status": "VERIFIED"
                    }
                    self.processed_persons[track_id] = person_data

                    new_event.append({
                        "track_id": track_id,
                        "bbox": bbox,
                        **person_data
                    })
                except Exception as e:
                    print(f"[DetectionService Error] Track ID {track_id}: {e}")


            cached_data = self.processed_persons.get(track_id, {
                "dress": None,
                "id_card": None,
                "snapshot": None,
                "status": "WAITING_IN_ROI" if not in_roi else "PROCESSING"
            })

            detected_persone_payload.append({
                "track_id": track_id,
                "bbox":bbox,
                "in_roi": in_roi,
                "dress": cached_data["dress"],
                "id_card": cached_data["id_card"],
                "snapshot": cached_data["snapshot"],
                "c_data": cached_data
            })
                
               


        
            
        return {
            "success": True,
            "camera_id": camera_id,
            "new_events": new_event,
            "active_tracks": detected_persone_payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
        }