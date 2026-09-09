from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO


class FoodDetector:
    """
    Team A - Food Detection

    Input:
        RGB numpy.ndarray

    Output:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str
            },
            ...
        ]

    Detection strategy:
        1. YOLO inference at 640
        2. If no detection >= confidence threshold,
           retry at 1280
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf: float = 0.25,
        primary_imgsz: int = 640,
        fallback_imgsz: int = 1280
    ):
        self.model = YOLO(str(model_path))

        self.conf = conf
        self.primary_imgsz = primary_imgsz
        self.fallback_imgsz = fallback_imgsz

    def _detect(
        self,
        image_bgr: np.ndarray,
        imgsz: int
    ):
        """Run one YOLO inference pass."""

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            imgsz=imgsz,
            verbose=False
        )

        height, width = image_bgr.shape[:2]

        detections = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                # Keep bounding box inside image
                x1 = max(
                    0,
                    min(int(round(x1)), width - 1)
                )

                y1 = max(
                    0,
                    min(int(round(y1)), height - 1)
                )

                x2 = max(
                    0,
                    min(int(round(x2)), width)
                )

                y2 = max(
                    0,
                    min(int(round(y2)), height)
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                detections.append({
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "confidence":
                        round(confidence, 4),

                    "class_id":
                        class_id,

                    "class_name":
                        self.model.names[class_id]
                })

        # Highest-confidence detection first
        detections.sort(
            key=lambda item:
                item["confidence"],
            reverse=True
        )

        return detections

    def predict(
        self,
        image_rgb: np.ndarray
    ):
        """
        Detect foods from an RGB image.
        """

        if not isinstance(
            image_rgb,
            np.ndarray
        ):
            raise TypeError(
                "Input must be an RGB numpy.ndarray."
            )

        if image_rgb.size == 0:
            raise ValueError(
                "Input image is empty."
            )

        if (
            image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
        ):
            raise ValueError(
                "Input must have shape (H, W, 3)."
            )

        # Contract of this wrapper is RGB.
        # Convert explicitly for our OpenCV/YOLO pipeline.
        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        # -------------------------
        # Primary inference: 640
        # -------------------------

        detections = self._detect(
            image_bgr,
            imgsz=self.primary_imgsz
        )

        if detections:
            return detections

        # -------------------------
        # Fallback inference: 1280
        # -------------------------

        detections = self._detect(
            image_bgr,
            imgsz=self.fallback_imgsz
        )

        return detections