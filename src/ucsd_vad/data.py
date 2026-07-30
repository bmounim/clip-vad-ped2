"""UCSD Pedestrian anomaly-detection dataset loading.

The UCSD Anomaly Detection Dataset ships as image sequences:

    UCSD_Anomaly_Dataset.v1p2/
        UCSDped2/
            Train/Train001/*.tif        normal clips only
            Test/Test001/*.tif          clips containing anomalies
            Test/Test001_gt/*.bmp       per-pixel anomaly masks

We follow the standard one-class protocol: training clips contain only
normal behaviour, and frame-level labels are needed for the test split
only. A test frame is labelled anomalous when its ground-truth mask
contains at least one positive pixel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FRAME_SUFFIXES = (".tif", ".tiff", ".jpg", ".png")
MASK_SUFFIXES = (".bmp", ".png", ".tif")


def _natural_key(path: Path) -> tuple:
    """Sort 001.tif, 002.tif, ..., 010.tif in numeric rather than lexical order."""
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _is_junk(path: Path) -> bool:
    """The published archive was packed on macOS and carries AppleDouble
    sidecars (``._001.tif``) and ``.DS_Store`` files. They share the image
    suffixes, so filtering on extension alone is not enough."""
    return path.name.startswith("._") or path.name.startswith(".")


def _list_images(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    images = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes and not _is_junk(p)
    ]
    return sorted(images, key=_natural_key)


def _list_frames(folder: Path) -> list[Path]:
    return _list_images(folder, FRAME_SUFFIXES)


@dataclass(frozen=True)
class Clip:
    """One video clip stored as an ordered list of frame paths.

    ``labels`` is None for training clips (normal by construction) and a
    0/1 array of length ``len(frames)`` for test clips.
    """

    name: str
    frames: list[Path]
    labels: np.ndarray | None

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def n_anomalous(self) -> int:
        return 0 if self.labels is None else int(self.labels.sum())


def find_dataset_root(search_root: Path) -> Path:
    """Locate the extracted ``UCSD_Anomaly_Dataset.v1p2`` directory.

    The archive nests unpredictably depending on how it was extracted, so
    we search rather than assume a fixed depth.
    """
    search_root = Path(search_root)
    if (search_root / "UCSDped2").is_dir():
        return search_root
    for candidate in sorted(search_root.rglob("UCSDped2")):
        if candidate.is_dir():
            return candidate.parent
    raise FileNotFoundError(
        f"Could not find a UCSDped2 folder under {search_root}. "
        "Run scripts/prepare_data.py first."
    )


def _mask_folder_for(test_clip: Path) -> Path | None:
    """Return the ``*_gt`` folder matching a test clip, if it exists."""
    gt = test_clip.parent / f"{test_clip.name}_gt"
    return gt if gt.is_dir() else None


def _labels_from_masks(mask_folder: Path, n_frames: int) -> np.ndarray:
    """Derive frame-level labels: a frame is anomalous if any mask pixel is set."""
    masks = _list_images(mask_folder, MASK_SUFFIXES)
    if not masks:
        raise ValueError(f"No mask images found in {mask_folder}")
    if len(masks) != n_frames:
        raise ValueError(
            f"{mask_folder.name}: {len(masks)} masks for {n_frames} frames. "
            "Refusing to guess the alignment."
        )
    labels = np.zeros(n_frames, dtype=np.int64)
    for i, mask_path in enumerate(masks):
        with Image.open(mask_path) as im:
            labels[i] = int(np.any(np.asarray(im) > 0))
    return labels


def load_split(
    dataset_root: Path, subset: str = "UCSDped2", split: str = "Test"
) -> list[Clip]:
    """Load every clip of a split.

    Args:
        dataset_root: directory containing ``UCSDped1``/``UCSDped2``.
        subset: ``"UCSDped1"`` or ``"UCSDped2"``.
        split: ``"Train"`` or ``"Test"``.

    Returns:
        Clips in natural order. Test clips carry frame-level labels.
    """
    split_dir = Path(dataset_root) / subset / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    clip_dirs = [
        d
        for d in sorted(split_dir.iterdir(), key=_natural_key)
        if d.is_dir() and not d.name.endswith("_gt") and not _is_junk(d)
    ]

    clips: list[Clip] = []
    for clip_dir in clip_dirs:
        frames = _list_frames(clip_dir)
        if not frames:
            continue
        labels = None
        if split.lower() == "test":
            mask_folder = _mask_folder_for(clip_dir)
            if mask_folder is None:
                raise FileNotFoundError(
                    f"No ground-truth folder for {clip_dir.name}; frame-level "
                    "evaluation would be impossible."
                )
            labels = _labels_from_masks(mask_folder, len(frames))
        clips.append(Clip(name=clip_dir.name, frames=frames, labels=labels))

    if not clips:
        raise ValueError(f"No clips found in {split_dir}")
    return clips


def describe(clips: list[Clip]) -> str:
    """Human-readable summary used in logs and the README."""
    total = sum(len(c) for c in clips)
    anomalous = sum(c.n_anomalous for c in clips)
    lines = [f"{len(clips)} clips, {total} frames"]
    if anomalous:
        lines.append(f"{anomalous} anomalous frames ({100 * anomalous / total:.1f}%)")
    return " | ".join(lines)
