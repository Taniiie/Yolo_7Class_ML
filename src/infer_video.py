from ultralytics import YOLO
import cv2

MODEL_PATH = "runs/detect/7class_run/weights/best.pt"
model = YOLO(MODEL_PATH)

video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640)[0]

    annotated = results.plot()
    cv2.imshow("Detections", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
