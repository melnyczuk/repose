# "https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#pose-estimation"  # noqa: E501

import csv
from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

NAMES: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


@dataclass(eq=True, frozen=True)
class PoseKeyPoint:
    x: float
    y: float
    name: str
    score: Optional[float] = None


@dataclass(eq=True, frozen=True)
class Coco:
    keypoints: list[PoseKeyPoint]
    score: Optional[float] = None

    LENGTH: int = 17 * 2  # aka 17 coordinates

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Coco":
        return cls(
            keypoints=[
                PoseKeyPoint(name=NAMES[i], x=x, y=y)
                for i, (x, y) in enumerate(arr.reshape((17, 2)))
            ]
        )

    def to_array(self: "Coco") -> np.ndarray:
        coords = ((keypoint.x, keypoint.y) for keypoint in self.keypoints)
        return np.array(list(coords)).flatten()


class CocoDataset(Dataset):
    rows: np.ndarray

    def __init__(
        self: "CocoDataset",
        path: str,
    ) -> None:
        with open(path, "r", newline="\n") as training_data_file:
            reader = csv.reader(training_data_file)
            rows = [tuple(map(float, row)) for row in reader]
        self.rows = np.array(rows, dtype=np.float32)

    def __len__(self: "CocoDataset") -> int:
        return len(self.rows)

    def __getitem__(self: "CocoDataset", idx: int) -> np.ndarray:
        return self.rows[idx]
