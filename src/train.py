from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8s.pt")  # you can use yolov8n.pt for faster training

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # GPU
        name="7class_run",
    )


if __name__ == "__main__":
    main()
