import base64
import binascii
import os
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator

CAMERA_ID_PATTERN = r"^[A-Za-z0-9_-]{1,48}$"
CALIBRATION_DIR = Path(os.getenv("CAMERA_CALIBRATION_DIR", "./calibrations")).resolve()

Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Point = tuple[Coordinate, Coordinate]


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def side(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return all(min(start, end) <= value <= max(start, end) for start, end, value in zip(p, q, r))

    ab_c, ab_d = side(a, b, c), side(a, b, d)
    cd_a, cd_b = side(c, d, a), side(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    return (
        (ab_c == 0 and on_segment(a, b, c))
        or (ab_d == 0 and on_segment(a, b, d))
        or (cd_a == 0 and on_segment(c, d, a))
        or (cd_b == 0 and on_segment(c, d, b))
    )


class CameraCalibration(BaseModel):
    frame_width: int = Field(gt=0, le=25_000_000)
    frame_height: int = Field(gt=0, le=25_000_000)
    points: list[Point] = Field(min_length=3, max_length=256)
    reference_frame_data_url: str | None = Field(default=None, max_length=4_000_000)

    @model_validator(mode="after")
    def validate_boundary(self) -> "CameraCalibration":
        points = self.points
        if len(set(points)) != len(points):
            raise ValueError("Точки границы должны отличаться")
        edges = list(zip(points, points[1:] + points[:1]))
        area_twice = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)
        if abs(area_twice) < 1e-6:
            raise ValueError("Контур должен ограничивать область трассы")
        for i, (a, b) in enumerate(edges):
            for j in range(i + 2, len(edges)):
                if i == 0 and j == len(edges) - 1:
                    continue
                if _segments_intersect(a, b, *edges[j]):
                    raise ValueError("Граница трассы не должна пересекать сама себя")
        if self.reference_frame_data_url is not None:
            self.reference_frame()
        return self

    def matches_frame(self, width: int, height: int) -> bool:
        scale = (width * self.frame_height) / (height * self.frame_width)
        return abs(scale - 1) <= 0.005

    def reference_frame(self) -> np.ndarray | None:
        if self.reference_frame_data_url is None:
            return None
        header, _, payload = self.reference_frame_data_url.partition(",")
        if header != "data:image/jpeg;base64":
            raise ValueError("Опорный кадр должен быть в формате JPEG")
        try:
            with Image.open(BytesIO(base64.b64decode(payload, validate=True))) as image:
                if image.format != "JPEG" or image.width * image.height > 25_000_000:
                    raise ValueError("Недопустимый формат или размер опорного кадра")
                if not self.matches_frame(image.width, image.height):
                    raise ValueError("Пропорции опорного кадра и разметки должны совпадать")
                return np.asarray(image.convert("L"))
        except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Не удалось прочитать опорный кадр") from exc

    def track_mask(self) -> dict:
        return {
            "model_name": "track_camera_calibration",
            "class_id": 0,
            "class_name": "track",
            "points": [list(point) for point in self.points],
        }


def outline_from_mask(mask: np.ndarray, *, main_contour: bool = False) -> list[Point]:
    """Convert a painted region into an editable camera boundary."""
    height, width = mask.shape
    mode = cv2.RETR_EXTERNAL if main_contour else cv2.RETR_LIST
    contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Закрась область трассы перед сохранением")
    if not main_contour and len(contours) != 1:
        raise ValueError("Закрашенная зона должна быть цельной. Соедини участки кистью и закрась отверстия.")

    contour = max(contours, key=cv2.contourArea)
    scale = np.array([max(width - 1, 1), max(height - 1, 1)])
    for tolerance in (0.8, 1.5, 2.5, 4.0):
        points = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2) / scale
        if not 3 <= len(points) <= 256:
            continue
        try:
            return CameraCalibration(frame_width=width, frame_height=height, points=points.tolist()).points
        except ValidationError:
            continue
    raise ValueError("Не удалось построить простой контур. Сгладь края закрашенной области.")


def load_calibration(camera_id: str) -> CameraCalibration | None:
    path = CALIBRATION_DIR / f"{camera_id}.json"
    try:
        return CameraCalibration.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def save_calibration(camera_id: str, calibration: CameraCalibration) -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=CALIBRATION_DIR, delete=False) as file:
        temporary_path = Path(file.name)
    try:
        temporary_path.write_text(calibration.model_dump_json(indent=2), encoding="utf-8")
        temporary_path.replace(CALIBRATION_DIR / f"{camera_id}.json")
    finally:
        temporary_path.unlink(missing_ok=True)
