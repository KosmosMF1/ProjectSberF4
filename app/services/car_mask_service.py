import os

import numpy as np
from dotenv import load_dotenv

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional CV dependency
    YOLO = None

load_dotenv()


class CarSegmentationService:
    """Adapter for a YOLO detection/segmentation model that predicts the whole car.

    The uploaded `best_cars.pt` is a YOLO segmentation checkpoint with one class:
    `cars`. The service also works with a detection-only YOLO model: if masks are
    absent, it converts the bounding box into a rectangle polygon so the existing
    frontend renderer can draw it.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        class_names: set[str] | None = None,
    ) -> None:
        self.model_path = model_path or os.getenv("CAR_MODEL_PATH", "")
        self.conf = float(conf if conf is not None else os.getenv("CAR_MODEL_CONF", "0.25"))
        self.iou = float(iou if iou is not None else os.getenv("CAR_MODEL_IOU", "0.7"))
        env_class_names = os.getenv("CAR_CLASS_NAMES", "car,cars,vehicle,formula,formula4")
        self.class_names = class_names or {
            item.strip().lower()
            for item in env_class_names.split(",")
            if item.strip()
        }
        self._model = None

    def _load_model(self):
        if not self.model_path or YOLO is None:
            return None
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"CAR_MODEL_PATH does not exist: {self.model_path}")
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    @staticmethod
    def _norm_points(points_xy: np.ndarray, width: int, height: int) -> list[list[float]]:
        return [
            [
                round(float(x) / float(width), 6),
                round(float(y) / float(height), 6),
            ]
            for x, y in points_xy
        ]

    @staticmethod
    def _bbox_to_polygon(xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
        return np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _norm_bbox(xyxy: np.ndarray, width: int, height: int) -> list[float]:
        x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
        return [
            round(max(0.0, min(1.0, x1 / float(width))), 6),
            round(max(0.0, min(1.0, y1 / float(height))), 6),
            round(max(0.0, min(1.0, x2 / float(width))), 6),
            round(max(0.0, min(1.0, y2 / float(height))), 6),
        ]

    @staticmethod
    def _to_yolo_segmentation(
        class_id: int,
        points_xy: np.ndarray,
        width: int,
        height: int,
    ) -> list[float]:
        segmentation: list[float] = [float(class_id)]
        for x, y in points_xy:
            segmentation.extend(
                [
                    round(float(x) / float(width), 6),
                    round(float(y) / float(height), 6),
                ]
            )
        return segmentation

    def _is_car_class(
        self,
        cls_id: int,
        cls_name: str,
        names_count: int,
    ) -> bool:
        normalized_name = cls_name.strip().lower()
        if normalized_name in self.class_names:
            return True
        # Single-class checkpoints often use class id 0 even if the class name
        # differs slightly between training/export environments.
        return names_count <= 1 and cls_id == 0

    def build_car_mask_payload(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, object]]:
        del frame_index

        height, width = frame_bgr.shape[:2]
        model = self._load_model()
        if model is None:
            return []

        results = model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            retina_masks=True,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        cls_names = getattr(result, "names", {}) or {}
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()

        polygons = []
        if result.masks is not None:
            polygons = list(result.masks.xy)

        payload: list[dict[str, object]] = []
        for index, xyxy in enumerate(boxes_xyxy):
            cls_id = int(class_ids[index]) if index < len(class_ids) else 0
            source_class_name = str(cls_names.get(cls_id, "car"))
            if not self._is_car_class(cls_id, source_class_name, len(cls_names)):
                continue

            if index < len(polygons) and polygons[index] is not None and len(polygons[index]) >= 3:
                poly_arr = np.asarray(polygons[index], dtype=np.float32)
            else:
                poly_arr = self._bbox_to_polygon(np.asarray(xyxy, dtype=np.float32))

            confidence = float(confidences[index]) if index < len(confidences) else 0.0

            payload.append(
                {
                    "model_name": "car_seg_yolo",
                    # Unified ids used in the frontend:
                    # 0 - track, 1 - wheel, 2 - violation, 3 - car.
                    "class_id": 3,
                    "class_name": "car",
                    "source_class_id": cls_id,
                    "source_class_name": source_class_name,
                    "confidence": round(confidence, 6),
                    "bbox_xyxy": self._norm_bbox(np.asarray(xyxy, dtype=np.float32), width, height),
                    "points": self._norm_points(poly_arr, width, height),
                    "yolo_segmentation": self._to_yolo_segmentation(
                        3,
                        poly_arr,
                        width,
                        height,
                    ),
                }
            )

        return payload
