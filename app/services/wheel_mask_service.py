import os

import numpy as np
from dotenv import load_dotenv

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional CV dependency
    YOLO = None

load_dotenv()


class WheelSegmentationService:
    """
    Adapter for a trained YOLO segmentation model that predicts wheel masks.

    If the model is unavailable, the service returns deterministic demo polygons
    so the UI and API contract remain usable in development environments.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        class_name: str = "wheel",
    ) -> None:
        self.model_path = model_path or os.getenv("WHEEL_MODEL_PATH", "")
        self.conf = conf
        self.iou = iou
        self.class_name = class_name
        self._model = None

    def _load_model(self):
        if not self.model_path or YOLO is None:
            return None
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    @staticmethod
    def _norm_points(points_xy: np.ndarray, width: int, height: int) -> list[list[float]]:
        return [
            [round(float(x) / float(width), 6), round(float(y) / float(height), 6)]
            for x, y in points_xy
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
                [round(float(x) / float(width), 6), round(float(y) / float(height), 6)]
            )
        return segmentation

    @staticmethod
    def _fallback_wheel_polygons(frame_index: int) -> list[np.ndarray]:
        t = frame_index % 120
        x0 = 0.35 + 0.0025 * t
        y0 = 0.62
        size = 0.045
        left = np.array(
            [
                [x0, y0],
                [x0 + size, y0],
                [x0 + size, y0 + size],
                [x0, y0 + size],
            ],
            dtype=np.float32,
        )

        x1 = 0.55 + 0.0020 * t
        right = np.array(
            [
                [x1, y0 - 0.01],
                [x1 + size, y0 - 0.01],
                [x1 + size, y0 - 0.01 + size],
                [x1, y0 - 0.01 + size],
            ],
            dtype=np.float32,
        )
        return [left, right]

    def _fallback_payload(
        self,
        frame_index: int,
        width: int,
        height: int,
    ) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for poly_norm in self._fallback_wheel_polygons(frame_index):
            poly_px = np.stack(
                [poly_norm[:, 0] * width, poly_norm[:, 1] * height],
                axis=1,
            )
            payload.append(
                {
                    "model_name": "wheel_segmentation_stub",
                    "class_id": 1,
                    "class_name": self.class_name,
                    "points": [
                        [round(float(x), 6), round(float(y), 6)]
                        for x, y in poly_norm.tolist()
                    ],
                    "yolo_segmentation": self._to_yolo_segmentation(
                        1,
                        poly_px,
                        width,
                        height,
                    ),
                }
            )
        return payload

    def build_wheel_mask_payload(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, object]]:
        height, width = frame_bgr.shape[:2]
        model = self._load_model()
        if model is None:
            return self._fallback_payload(frame_index, width, height)

        payload: list[dict[str, object]] = []
        results = model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            retina_masks=True,
        )
        if not results:
            return payload

        result = results[0]
        if result.masks is None or result.boxes is None:
            return payload

        cls_names = getattr(result, "names", {}) or {}
        polygons = result.masks.xy
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        for i, poly in enumerate(polygons):
            if poly is None or len(poly) < 3:
                continue
            cls_id = int(class_ids[i]) if i < len(class_ids) else 1
            cls_name = str(cls_names.get(cls_id, self.class_name))
            if cls_name != self.class_name and cls_id != 1:
                continue

            poly_arr = np.asarray(poly, dtype=np.float32)
            payload.append(
                {
                    "model_name": "wheel_seg_yolo",
                    "class_id": 1,
                    "class_name": self.class_name,
                    "points": self._norm_points(poly_arr, width, height),
                    "yolo_segmentation": self._to_yolo_segmentation(
                        1,
                        poly_arr,
                        width,
                        height,
                    ),
                }
            )

        return payload
