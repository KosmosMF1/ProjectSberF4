import os

import numpy as np
from dotenv import load_dotenv

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional CV dependency
    YOLO = None

load_dotenv()


class TrackSegmentationService:
    """
    Adapter for a trained YOLO segmentation model that predicts the track mask.

    The service returns normalized polygons in the same payload format as wheel
    segmentation, so API clients can render masks independently of image size.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        class_name: str = "track",
        allow_fallback: bool = False,
    ) -> None:
        self.model_path = model_path or os.getenv("TRACK_MODEL_PATH", "")
        self.conf = conf
        self.iou = iou
        self.class_name = class_name
        self.allow_fallback = allow_fallback
        self._model = None

    def _load_model(self):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed; track model cannot be loaded")
        if not self.model_path:
            raise RuntimeError("TRACK_MODEL_PATH is not configured")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"TRACK_MODEL_PATH does not exist: {self.model_path}")
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
    def _fallback_track_polygon() -> np.ndarray:
        return np.array(
            [
                [0.04, 0.16],
                [0.94, 0.12],
                [0.98, 0.88],
                [0.03, 0.92],
            ],
            dtype=np.float32,
        )

    def _fallback_payload(self, width: int, height: int) -> list[dict[str, object]]:
        poly_norm = self._fallback_track_polygon()
        poly_px = np.stack(
            [poly_norm[:, 0] * width, poly_norm[:, 1] * height],
            axis=1,
        )
        return [
            {
                "model_name": "track_segmentation_stub",
                "class_id": 0,
                "class_name": self.class_name,
                "points": [
                    [round(float(x), 6), round(float(y), 6)]
                    for x, y in poly_norm.tolist()
                ],
                "yolo_segmentation": self._to_yolo_segmentation(0, poly_px, width, height),
            }
        ]

    def build_track_mask_payload(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, object]]:
        del frame_index

        height, width = frame_bgr.shape[:2]
        try:
            model = self._load_model()
        except RuntimeError:
            if self.allow_fallback:
                return self._fallback_payload(width, height)
            raise

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

        polygons = result.masks.xy

        for poly in polygons:
            if poly is None or len(poly) < 3:
                continue

            poly_arr = np.asarray(poly, dtype=np.float32)
            payload.append(
                {
                    "model_name": "track_seg_yolo",
                    "class_id": 0,
                    "class_name": self.class_name,
                    "points": self._norm_points(poly_arr, width, height),
                    "yolo_segmentation": self._to_yolo_segmentation(
                        0,
                        poly_arr,
                        width,
                        height,
                    ),
                }
            )

        return payload
