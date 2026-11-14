"""Ultralytics YOLO object detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config

@dataclass(frozen=True)
class Detection:
    """A detected object's image-space bounding box and label."""

    xyxy: tuple[int, int, int, int]
    label: str
    confidence: float


class YOLODetector:
    """Run a closed-set or open-vocabulary Ultralytics detector."""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        device: str | None = None,
        detector_type: str = config.DETECTOR_TYPE,
        vocabulary: tuple[str, ...] = config.OPEN_VOCABULARY_CLASSES,
    ) -> None:
        try:
            from ultralytics import YOLO

        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Install dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from error

        if detector_type == "coco":
            self.model = YOLO(model_path)
        elif detector_type == "world":
            if not vocabulary or any(not item.strip() for item in vocabulary):
                raise ValueError("open-vocabulary detection requires non-empty text classes")
            try:
                from ultralytics import YOLOWorld
            except ImportError as error:
                raise RuntimeError(
                    "This Ultralytics version does not provide YOLOWorld. "
                    "Upgrade ultralytics to use the open-vocabulary detector."
                ) from error
            self.model = YOLOWorld(model_path)
            self.model.set_classes(list(vocabulary))
        else:
            raise ValueError(f"unsupported detector type: {detector_type}")
        self.confidence = confidence
        self.device = device
        self.detector_type = detector_type

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one BGR frame."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                names = result.names
                label = (
                    names.get(class_id, str(class_id))
                    if isinstance(names, dict)
                    else names[class_id]
                )
                detections.append(
                    Detection(
                        xyxy=(x1, y1, x2, y2),
                        label=label,
                        confidence=float(box.conf[0].item()),
                    )
                )
        return detections
