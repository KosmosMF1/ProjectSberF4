import base64

import cv2
import numpy as np


def encode_bgr_to_data_url(frame_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def encode_mask_to_data_url(mask: np.ndarray) -> str:
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    ok, buffer = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("Failed to encode mask as PNG")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"
