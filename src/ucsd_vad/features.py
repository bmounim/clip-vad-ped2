"""CLIP feature extraction with on-disk caching.

Both methods in this repository are training-free and operate entirely on
frozen CLIP embeddings, so extraction runs exactly once per (backbone,
subset, split) and every experiment and ablation reads the cache. On CPU
this is the dominant cost; caching turns a full ablation sweep from hours
into seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .data import Clip

DEFAULT_BACKBONE = "openai/clip-vit-base-patch32"


def _as_embedding(output) -> torch.Tensor:
    """Normalise the return type of ``get_*_features`` across transformers versions.

    transformers <5 returns a plain tensor; transformers >=5 returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds the same
    projected embedding (verified numerically identical to running
    ``visual_projection(vision_model(...).pooler_output)`` by hand).
    """
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise TypeError(
            f"Cannot extract an embedding tensor from {type(output).__name__}"
        )
    return pooled


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEncoder:
    """Thin wrapper over a frozen CLIP model producing L2-normalised embeddings."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, device: str = "auto"):
        # Imported lazily so that `import ucsd_vad` stays cheap.
        from transformers import CLIPModel, CLIPProcessor

        self.backbone = backbone
        self.device = resolve_device(device)
        self.model = CLIPModel.from_pretrained(backbone).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(backbone)

    @torch.no_grad()
    def encode_images(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """Encode frames to a (N, D) array of unit-norm embeddings."""
        out: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            # UCSD frames are single-channel TIFFs; CLIP expects 3-channel RGB.
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            feats = _as_embedding(self.model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype(np.float32))
            for im in images:
                im.close()
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode prompts to a (T, D) array of unit-norm embeddings."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        feats = _as_embedding(self.model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def cache_path(cache_dir: Path, backbone: str, subset: str, split: str) -> Path:
    tag = backbone.replace("/", "__")
    return Path(cache_dir) / f"{subset}_{split}_{tag}.npz"


def extract_split(
    clips: list[Clip],
    encoder: ClipEncoder,
    cache_file: Path,
    batch_size: int = 32,
    overwrite: bool = False,
) -> tuple[dict[str, np.ndarray], bool]:
    """Encode every clip of a split, caching the result as a single npz.

    Returns ``(features, computed_fresh)`` where features maps
    ``clip_name -> (n_frames, D)``. The flag matters for timing: a warm
    cache would otherwise be reported as an absurd encoding throughput.
    """
    cache_file = Path(cache_file)
    if cache_file.exists() and not overwrite:
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}, False

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    features: dict[str, np.ndarray] = {}
    for clip in tqdm(clips, desc=f"encoding {cache_file.stem}", unit="clip"):
        features[clip.name] = encoder.encode_images(clip.frames, batch_size=batch_size)
    np.savez_compressed(cache_file, **features)
    return features, True


def measure_encode_fps(
    encoder: ClipEncoder, paths: list[Path], batch_size: int = 32, warmup: int = 8
) -> float:
    """Measure end-to-end encoding throughput on real frames.

    Reported independently of the feature cache so the figure stays
    truthful across reruns. Includes image decoding and preprocessing,
    which is what an actual deployment pays for.
    """
    import time

    if len(paths) <= warmup:
        raise ValueError("Need more frames than the warm-up count")
    encoder.encode_images(paths[:warmup], batch_size=batch_size)
    timed = paths[warmup:]
    start = time.perf_counter()
    encoder.encode_images(timed, batch_size=batch_size)
    return len(timed) / (time.perf_counter() - start)
