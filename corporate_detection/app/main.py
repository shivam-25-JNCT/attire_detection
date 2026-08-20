import os
import json
from datetime import datetime, timezone
import cv2
from dotenv import load_dotenv

# Load models and services

from detect_service.person_detection import PersoneDetector
from detect_service.id_card_detection import IDCardDetector
from classification.dress_detect import DressClassifier
from service.detection_service import DetectionService
from utils.image import crop_persone,save_crop_snapshot
# from services.s3_service import S3Service
from config import settings

load_dotenv()

# AWS / S3 Config from 
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "my-attendance-bucket")

VIDEO_SOURCE = settings.DEFAULT_RTSP_URL

if VIDEO_SOURCE.isdigit():
    VIDEO_SOURCE = int(VIDEO_SOURCE)



def main():
    print("--- Loading Models into Memory ---")
    person_det = PersoneDetector(
        model_path=settings.PERSON_MODEL_PATH,
        conf_thresh=settings.PERSON_CONF_THRESHOLD,
        device=settings.DEVICE
    )
    id_card_det = IDCardDetector(
        model_path=settings.ID_CARD_MODEL_PATH,
        conf_thresh=settings.ID_CARD_CONF_THRESHOLD,
        device=settings.DEVICE
    )
    dress_cls = DressClassifier(
        model_name=settings.CLIP_MODEL_PATH,
        device=settings.DEVICE
    )

    det_service = DetectionService(
        person_detector=person_det,
        id_card_detector=id_card_det,
        dress_claassifier=dress_cls
    )

    # s3_service = S3Service(
    #     bucket_name=S3_BUCKET,
    #     aws_access_key=AWS_KEY,
    #     aws_secret_key=AWS_SECRET,
    #     region_name=AWS_REGION
    # )

    print(f"--- Opening Video Stream: {VIDEO_SOURCE} ---")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    frame_count = 0
    frame_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #  ROI
    
    roi_x1=int(frame_w*0.10)
    roi_y1=int(frame_h*0.15)
    roi_x2=int(frame_w*0.90)
    roi_y2=int(frame_h*0.85)
    ROI_ZONE=(roi_x1,roi_y1,roi_x2,roi_y2)

    # cropped_tracks = set()
    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or frame not readable.")
            break

        frame_count += 1
        result = det_service.process_frame(frame, camera_id="CAM_TEST_01",roi=ROI_ZONE)


        if result["new_events"]:
            for evt in result["new_events"]:
                print("\n================ DETECTED JSON ================")
                print("\n[NEW PERSON VERIFIED EVENT]")
                print(json.dumps(evt, indent=2))
                print("===============================================\n")

            
        for person in result["active_tracks"]:
            track_id=person["track_id"]
            bbox=person["bbox"]
            data=person["c_data"]
            x1, y1, x2, y2 = bbox

            if data["status"]=="VERIFIED":
                dress_info = data.get("dress") or {}
                # dress_label = data["dress"]["lable"]
                dress_label = dress_info.get("label") or dress_info.get("lable", "UNKNOWN")

                id_info = data.get("id_card")
                if isinstance(id_info, dict):
                    id_detected = id_info.get("detected", False)
                else:
                    id_detected = bool(id_info)
                
                # color = (0, 255, 0) if (id_detected and dress_label.lower() == "formal") else (0, 0, 255)

                id_color = (0,255,0) if (id_detected) else (0,0,255)
                formal_color = (0,255,0) if (dress_label.lower() == "formal") else (0,0,255)

                start_x=x1
                text_y=max(20,y1-8)
                # track id 
                id_text=f"ID:{track_id} | "
                cv2.putText(
                        frame,
                        id_text,
                        (start_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
                #get width of tract id 
                (id_w, id_h),baseline  = cv2.getTextSize(
                        id_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        2
                    )
                # for dress 
                dress_text=f"{str(dress_label).upper()} | " 
                cv2.putText(frame,dress_text,(start_x + id_w, text_y),cv2.FONT_HERSHEY_SIMPLEX,0.55,formal_color,2)

                ( dress_w, dress_h),baseline = cv2.getTextSize(
                    dress_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2
                )
                #  for id card
                card_text = f"CARD:{id_detected}"

                cv2.putText(
                    frame,
                    card_text,
                    (start_x + id_w + dress_w, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    id_color,
                    2
                )
                                

                # text = f"ID:{track_id} | {str(dress_label).upper()} | CARD:{id_detected}"
                timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                # s3_filename = f"detections/CAM_TEST_01/{timestamp_str}.json"
                # s3_service.upload_json(data=result, s3_key=s3_filename)

               
            else:
                color=(200,200,200)
                text=f"Tracting ID : {track_id}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            # cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # draw roi
    

        cv2.rectangle(frame, (roi_x1,roi_y1), (roi_x2, roi_y2),(255,255,0), 2)
        cv2.putText(frame, "DETECTION ZONE (ROI)", (roi_x1+10,roi_y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

       
        cv2.imshow("Real-Time Corporate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()