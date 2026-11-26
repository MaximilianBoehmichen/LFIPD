import random
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import torch
from jaxtyping import UInt8
from PIL import Image
from torch.utils.data import Dataset

UPENN_Slice = UInt8[torch.Tensor, "240 240"]


class EvalDataset(Dataset):
    """The Dataset of samples with all modalities."""

    sample_dirs: list[Path]
    slice_counts: list[int]
    slice_offsets: np.ndarray
    transforms: Callable[[dict[str, UPENN_Slice]], dict[str, UPENN_Slice]]

    SPLIT_PERCENTAGE: ClassVar[dict[str, float]] = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
    }
    MODALITIES: ClassVar[list[str]] = [
        "FLAIR",
        "T1",
        "T1GD",
        "T2",
    ]

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        root: Path = Path("../../data/dataset"),
        transforms: Callable[
            [dict[str, UPENN_Slice]], dict[str, UPENN_Slice]
        ] = lambda x: x,
    ) -> None:
        random.seed(848577)
        root = root / "eval" / "images"

        samples = sorted(f for f in root.iterdir() if f.is_dir())
        k = int(len(samples) * self.SPLIT_PERCENTAGE[split])

        self.sample_dirs = random.sample(samples, k)
        self.transforms = transforms

        self.slice_counts = [
            len(list((d / "seg").glob("*.png"))) for d in self.sample_dirs
        ]
        self.slice_offsets = np.concatenate(([0], np.cumsum(self.slice_counts)[:-1]))

    def __len__(self) -> int:
        return int(sum(self.slice_counts))

    def __getitem__(self, index: int) -> dict[str, UPENN_Slice]:
        patient = int(np.searchsorted(self.slice_offsets, index, side="right")) - 1
        local_index = index - int(self.slice_offsets[patient])

        sample_dir = self.sample_dirs[patient]
        slice_name = sorted((sample_dir / "seg").glob("*.png"))[local_index].name

        data = {
            **{
                modality: self._load_slice(sample_dir / modality / slice_name)
                for modality in self.MODALITIES
                if (sample_dir / modality).exists()
            },
            "label": self._load_slice(sample_dir / "seg" / slice_name),
        }

        return self.transforms(data)

    @staticmethod
    def _load_slice(path: Path) -> UPENN_Slice:
        """Load a single 2D slice as a uint8 tensor."""
        return torch.from_numpy(np.array(Image.open(path), dtype=np.uint8))
