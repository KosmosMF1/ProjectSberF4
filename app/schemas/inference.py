from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class YoloMask(BaseModel):
    model_name: str = Field(..., description="Model alias that produced the mask")
    class_id: int = Field(..., description="Unified class id used by the web app")
    class_name: str = Field(..., description="Human readable class name")
    points: list[list[float]] = Field(
        ..., description="Normalized polygon points as [x, y] pairs"
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_xyxy: list[float] | None = Field(
        default=None,
        description="Normalized bounding box as [x_min, y_min, x_max, y_max]",
    )


class ViolationRegion(BaseModel):
    model_name: str = Field(..., description="Model alias that produced the violation")
    class_id: int = Field(..., description="Violation class id")
    class_name: str = Field(..., description="Human readable violation class name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(
        ...,
        description="Normalized bounding box as [x_min, y_min, x_max, y_max]",
        min_length=4,
        max_length=4,
    )
    points: list[list[float]] = Field(
        ..., description="Normalized violation polygon points as [x, y] pairs"
    )
    yolo_segmentation: list[float] = Field(
        ..., description='YOLO segmentation line: "cls x1 y1 x2 y2 ..."'
    )


class FrameInferenceResponse(BaseModel):
    frame_index: int
    frame_width: int
    frame_height: int
    frame_data_url: str = Field(..., description="Frame encoded as data URL")
    masks: list[YoloMask]


class ViolationAnalysisRequest(BaseModel):
    frame_index: int = Field(ge=0)
    offtrack_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    hard_violation_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    track_prompts: list[str] | None = None


class WheelOfftrackDetail(BaseModel):
    instance_id: int
    outside_ratio: float
    outside_pixels: int
    total_pixels: int


class ViolationAnalysisResponse(BaseModel):
    frame_index: int
    frame_width: int
    frame_height: int
    frame_data_url: str
    annotated_frame_data_url: str | None
    track_mask_data_url: str | None
    violation_mask_data_url: str | None
    violation_detected: bool
    violation_score: float
    reason: str
    offtrack_wheels: list[WheelOfftrackDetail] | None
    violation_regions: list[ViolationRegion] | None
    masks: list[dict[str, Any]]


class VideoInferenceResponse(BaseModel):
    filename: str
    source_fps: float | None
    sample_fps: float
    frames_processed: int
    total_frames_estimate: int | None
    truncated: bool
    results: list[ViolationAnalysisResponse]


@dataclass(slots=True)
class OfftrackAnalysisResult:
    violation_detected: bool
    violation_score: float
    reason: str
    offtrack_wheels: list[WheelOfftrackDetail] | None
    annotated_frame_bgr: np.ndarray | None
    track_mask: np.ndarray | None
    violation_mask: np.ndarray | None


@dataclass(slots=True)
class TrackSegmentationResult:
    mask: np.ndarray  # bool array, HxW
    polygon_px: list[tuple[int, int]]
    prompt: str
    score: float


@dataclass(slots=True)
class WheelMaskResult:
    masks: list[dict[str, object]]
