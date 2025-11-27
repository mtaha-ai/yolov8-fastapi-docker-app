# detector.py
from typing import Dict, List
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from PIL import Image


class YoloObjectDetector:
    """
    Wrapper around a YOLOv8 model for object detection.
    Responsible for:
      - loading the model
      - running inference
      - returning a JSON-serializable detections dict
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
    ):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        model = YOLO(self.model_path)
        # device can be "cpu", "cuda", "mps"
        model.to(self.device)
        return model

    def _run_inference(self, image: Image.Image):
        """
        Run YOLO inference on a PIL image and return raw result object.
        """
        img_np = np.array(image)
        results = self.model(
            img_np,
            conf=self.conf_threshold,
            verbose=False
        )
        return results[0]  # single image result

    def _build_detections_dict(
        self,
        boxes: np.ndarray,
        cls_ids: np.ndarray,
        scores: np.ndarray,
        names: Dict[int, str],
        image_width: int,
        image_height: int,
    ) -> Dict:
        """
        Build a JSON-serializable detections dict.
        """
        detections: List[Dict] = []
        for box, cls_id, score in zip(boxes, cls_ids, scores):
            x1, y1, x2, y2 = box.tolist()
            class_id = int(cls_id)
            label = names.get(class_id, str(class_id))

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": label,
                    "confidence": float(score),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "x_center": float((x1 + x2) / 2.0),
                        "y_center": float((y1 + y2) / 2.0),
                        "image_width": image_width,
                        "image_height": image_height,
                    },
                }
            )

        return {
            "image_width": image_width,
            "image_height": image_height,
            "num_detections": len(detections),
            "detections": detections,
        }

    def predict(self, image: Image.Image) -> Dict:
        """
        Public method:
          - runs inference
          - returns a detections dict
        """
        if image is None:
            return {
                "image_width": 0,
                "image_height": 0,
                "num_detections": 0,
                "detections": [],
            }

        image = image.convert("RGB")
        result = self._run_inference(image)

        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        cls_ids = result.boxes.cls.cpu().numpy()  # (N,)
        scores = result.boxes.conf.cpu().numpy()  # (N,)
        names = result.names                     # id -> class_name

        w, h = image.size
        detections_dict = self._build_detections_dict(
            boxes=boxes,
            cls_ids=cls_ids,
            scores=scores,
            names=names,
            image_width=w,
            image_height=h,
        )
        return detections_dict