from ultralytics import YOLO
import cv2


class YOLODetector:
    def __init__(self) -> None:
        self.model = YOLO(r"D:\Projects\YOLO-7Class-Detection\runs\detect\7class_run\weights\best.pt")
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        results = self.model(frame, imgsz=640)[0]
        annotated = results.plot()

        return annotated
