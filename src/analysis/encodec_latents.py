"""Latent-space diagnostics for codec-style audio representations.

The production target is EnCodec/DAC token analysis. This module keeps the
core analysis dependency-light by accepting any frame-level latent matrix and
computing PCA plus label separability. The included WAV feature extractor is a
deterministic proxy for offline smoke tests; real EnCodec embeddings can be
plugged into the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True)
class LatentExample:
    path: str
    speaker: str = "unknown"
    content: str = "unknown"
    style: str = "unknown"
    environment: str = "unknown"


@dataclass(frozen=True)
class PCAResult:
    coordinates: np.ndarray
    explained_variance: np.ndarray
    components: np.ndarray


def load_manifest(path: Path) -> list[LatentExample]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            LatentExample(
                path=row["path"],
                speaker=row.get("speaker", "unknown") or "unknown",
                content=row.get("content", "unknown") or "unknown",
                style=row.get("style", "unknown") or "unknown",
                environment=row.get("environment", "unknown") or "unknown",
            )
            for row in reader
        ]


def wav_proxy_latent(path: str | Path, num_bands: int = 16) -> np.ndarray:
    """Return a compact acoustic latent vector from a WAV file.

    This is not a substitute for EnCodec. It is a zero-download proxy that lets
    the representation analysis run on checked-in audio.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()

    dtype = np.int16 if sample_width == 2 else np.uint8
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sample_width == 2:
        audio /= 32768.0
    else:
        audio = (audio - 128.0) / 128.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if len(audio) == 0:
        return np.zeros(num_bands + 4, dtype=np.float32)

    spectrum = np.abs(np.fft.rfft(audio))
    splits = np.array_split(spectrum, num_bands)
    bands = np.asarray([np.log1p(b.mean()) for b in splits], dtype=np.float32)
    stats = np.asarray(
        [audio.mean(), audio.std(), np.mean(np.abs(audio)), np.percentile(np.abs(audio), 95)],
        dtype=np.float32,
    )
    return np.concatenate([bands, stats])


def compute_pca(latents: np.ndarray, n_components: int = 2) -> PCAResult:
    if latents.ndim != 2:
        raise ValueError("latents must be a 2D array")
    if latents.shape[0] < 2:
        raise ValueError("at least two latent vectors are required")

    centered = latents - latents.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    coordinates = centered @ components.T
    variance = singular_values**2
    explained = variance[:n_components] / max(variance.sum(), 1e-12)
    return PCAResult(coordinates=coordinates, explained_variance=explained, components=components)


def label_separability(coordinates: np.ndarray, labels: list[str]) -> float:
    """Between-class variance / total variance in PCA space."""
    if len(labels) != len(coordinates):
        raise ValueError("labels and coordinates must have the same length")
    total = float(np.sum((coordinates - coordinates.mean(axis=0)) ** 2))
    if total == 0:
        return 0.0

    score = 0.0
    labels_arr = np.asarray(labels)
    for label in sorted(set(labels)):
        group = coordinates[labels_arr == label]
        score += len(group) * float(np.sum((group.mean(axis=0) - coordinates.mean(axis=0)) ** 2))
    return score / total


def analyze_manifest(examples: list[LatentExample], n_components: int = 2) -> dict:
    latents = np.stack([wav_proxy_latent(ex.path) for ex in examples])
    pca = compute_pca(latents, n_components=n_components)
    return {
        "n_examples": len(examples),
        "explained_variance": pca.explained_variance.tolist(),
        "speaker_separability": label_separability(pca.coordinates, [ex.speaker for ex in examples]),
        "content_separability": label_separability(pca.coordinates, [ex.content for ex in examples]),
        "style_separability": label_separability(pca.coordinates, [ex.style for ex in examples]),
        "environment_separability": label_separability(pca.coordinates, [ex.environment for ex in examples]),
    }
