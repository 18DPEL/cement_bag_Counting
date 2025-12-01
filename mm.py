# car_detection.py
import cv2
from ultralytics import YOLO


class CarDetector:
    """
    Handles YOLO model loading and car detection.
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_cars(self, frame):
        """
        Runs detection on the given frame.
        Returns list of detections -> [(x1, y1, x2, y2, conf, label), ...]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]

            if label.lower() in ["car", "truck", "bus"] and conf > self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2, conf, label))

        return detections
