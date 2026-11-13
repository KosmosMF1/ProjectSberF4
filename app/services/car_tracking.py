from dataclasses import dataclass
from math import hypot
from typing import Any

Mask = dict[str, Any]
Box = tuple[float, float, float, float]


def box_for(mask: Mask) -> Box:
    if box := mask.get("bbox_xyxy"):
        x1, y1, x2, y2 = box
    else:
        xs, ys = zip(*mask["points"])
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2, y2


def center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def overlap(left: Box, right: Box) -> tuple[float, float]:
    def area(box: Box) -> float:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    shared = area((max(left[0], right[0]), max(left[1], right[1]),
                   min(left[2], right[2]), min(left[3], right[3])))
    if not shared:
        return 0.0, 0.0
    left_area, right_area = area(left), area(right)
    return shared / (left_area + right_area - shared), shared / min(left_area, right_area)


def unique_masks(masks: list[Mask], *, iou: float, containment: float) -> list[Mask]:
    kept: list[Mask] = []
    boxes: list[Box] = []
    for mask in sorted(masks, key=lambda item: item.get("confidence") or 0, reverse=True):
        if len(mask.get("points", ())) < 3:
            continue
        box = box_for(mask)
        if any(a >= iou or b >= containment for a, b in (overlap(box, other) for other in boxes)):
            continue
        kept.append(mask)
        boxes.append(box)
    return kept


def assign_wheels(cars: list[Mask], wheels: list[Mask]) -> list[list[int]]:
    """Assign a wheel once, leaving ambiguous wheels between overlapping cars unassigned."""
    boxes = [box_for(car) for car in cars]
    assignments: list[list[int]] = [[] for _ in boxes]
    for index, wheel in enumerate(wheels):
        wx, wy = center(box_for(wheel))
        candidates = []
        for car_index, (x1, y1, x2, y2) in enumerate(boxes):
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            dx = abs(wx - (x1 + x2) / 2) / width
            dy = abs(wy - (y1 + y2) / 2) / height
            if dx <= 0.62 and dy <= 0.65:
                candidates.append((hypot(dx, dy), car_index))
        candidates.sort()
        if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.12):
            assignments[candidates[0][1]].append(index)
    return assignments


@dataclass
class Track:
    box: Box
    time: float
    velocity: tuple[float, float] = (0.0, 0.0)


class CarTracker:
    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, cars: list[Mask], timestamp: float) -> list[Mask]:
        self.tracks = {
            key: track for key, track in self.tracks.items() if timestamp - track.time <= 0.75
        }
        boxes = [box_for(car) for car in cars]
        matches = []
        for track_id, track in self.tracks.items():
            dt = timestamp - track.time
            dx, dy = (speed * dt for speed in track.velocity)
            x1, y1, x2, y2 = track.box
            predicted = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            px, py = center(predicted)
            for index, box in enumerate(boxes):
                x, y = center(box)
                iou, _ = overlap(predicted, box)
                sx = max(x2 - x1, box[2] - box[0], 0.015)
                sy = max(y2 - y1, box[3] - box[1], 0.015)
                distance = hypot((x - px) / sx, (y - py) / sy)
                if iou >= 0.05 or distance <= 0.85:
                    matches.append((distance + 1 - iou, track_id, index))

        assigned: dict[int, int] = {}
        used: set[int] = set()
        for _, track_id, index in sorted(matches):
            if track_id not in used and index not in assigned:
                assigned[index] = track_id
                used.add(track_id)

        result = []
        for index, (car, box) in enumerate(zip(cars, boxes, strict=True)):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                track = Track(box, timestamp)
            else:
                previous = self.tracks[track_id]
                dt = timestamp - previous.time
                x, y = center(box)
                px, py = center(previous.box)
                velocity = ((x - px) / dt, (y - py) / dt) if dt > 0 else previous.velocity
                track = Track(box, timestamp, velocity)
            self.tracks[track_id] = track
            result.append({**car, "track_id": track_id})
        return result
