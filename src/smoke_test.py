import cv2
from ultralytics import YOLO
from pathlib import Path

# Config
MODEL_PATH = "runs/detect/7class_run/weights/best.pt"
TEST_IMAGE = "datasets/images/val/000000000_vcluttered_hallway.png"

def test_inference():
    print(f"--- YOLO Smoke Test ---")
    
    # 1. Check Model
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return False
    print(f"SUCCESS: Model found.")

    # 2. Load Model
    try:
        model = YOLO(MODEL_PATH)
        print(f"SUCCESS: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False

    # 3. Check Image
    if not Path(TEST_IMAGE).exists():
        print(f"ERROR: Test image not found at {TEST_IMAGE}")
        return False
    print(f"SUCCESS: Test image found.")

    # 4. Run Inference
    try:
        results = model(TEST_IMAGE, imgsz=640)
        print(f"SUCCESS: Inference completed.")
        for result in results:
            print(f"Detected {len(result.boxes)} objects.")
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f" - Class {cls} with confidence {conf:.2f}")
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        return False

    print(f"--- Test Passed ---")
    return True

if __name__ == "__main__":
    test_inference()
