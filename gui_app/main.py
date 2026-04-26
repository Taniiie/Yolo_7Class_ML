import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
from yolo_worker import YOLODetector


class DetectionApp(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO 7-Class Detection")

        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.worker = YOLODetector()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self) -> None:
        frame = self.worker.get_frame()
        if frame is not None:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.label.setPixmap(QPixmap.fromImage(img))


app = QApplication(sys.argv)
win = DetectionApp()
win.show()
sys.exit(app.exec_())
