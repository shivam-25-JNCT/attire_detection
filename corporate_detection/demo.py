from ultralytics import YOLO
import cv2
import os
# Load model
model = YOLO(r"F:\person_detection\service\models\cloth_3.pt")

# Input image
image_path = r"F:\person_detection\service\img\formal.png"

# Run prediction
results = model(image_path)
val = 0
for result in results:

    
    if result.boxes is not None and len(result.boxes) > 0:

        for box in result.boxes:

            
            class_id = int(box.cls[0])

            
            confidence = float(box.conf[0])

            # Class name
            class_name = result.names[class_id]
            # if class_name == "" and confidence > 0.60:
            print(f"is formal suit = yes: {class_name}")
            print(f"Confidence: {confidence:.2f}")
          

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            #  for crop image and save image into the folder
            person_height = y2 - y1
            crop_y1=y1+int(person_height*0.2)
            crop_y2=y1+int(person_height*0.65)
            cropped_for_id_card= result.orig_img[crop_y1:crop_y2, x1:x2]
            cv2.imshow("neck_crop", cropped_for_id_card)
            folder=r"F:\person_detection\service\img\croped_img"

            os.makedirs(folder, exist_ok=True)
            output_path = os.path.join(folder, "cropped_for_id_card_{val}.jpg")
            cv2.imwrite(output_path, cropped_for_id_card)
            val+=1
            cv2.rectangle(
                result.orig_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                result.orig_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        # else:
        #     print("there is no Formal in this image")
    else:
        print("there is no formal suit in this image")

    # Show result
    cv2.imshow("Result", result.orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()