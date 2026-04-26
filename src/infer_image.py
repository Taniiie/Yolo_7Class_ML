from ultralytics import YOLO
import cv2
import sys

MODEL_PATH = "runs/detect/7class_run/weights/best.pt"


model = YOLO(MODEL_PATH)

img_path = sys.argv[1]

results = model(img_path)

results[0].show()  # displays result window
results[0].save(filename="output.jpg")

print("Saved output.jpg")
