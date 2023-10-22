import supervision as sv
from ultralytics import YOLO
import cv2


cap = cv2.VideoCapture("video.mp4")
model = YOLO("yolov8n.pt")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

    result = model.track(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > 0.5]

    if (detections.confidence > 0.5).any() == False:
        cv2.imshow('frame', frame)
        continue

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        result.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
    
    cv2.imshow('frame', annotated_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
