"""
occlusion_detector.py

Detector "inteligente" (heurístico) de oclusões (mão/braço) baseado em:
- diferença temporal entre frames (motion energy)
- normalização e limiarização adaptativa
- agrupamento de picos em eventos

Observação: não é um modelo de IA; é um detector robusto e simples, pronto
para evoluir (ex.: MediaPipe/YOLO opcional).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OcclusionEvent:
    start_frame: int
    end_frame: int
    score_peak: float
    kind: str = "occlusion"


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int


def _read_gray(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def _parse_roi(roi: Optional[str]) -> Optional[Roi]:
    if roi is None:
        return None
    try:
        parts = [int(p.strip()) for p in roi.split(",")]
    except ValueError as exc:
        raise ValueError("ROI inválida. Use o formato x,y,w,h (inteiros).") from exc
    if len(parts) != 4:
        raise ValueError("ROI inválida. Use o formato x,y,w,h (inteiros).")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("ROI inválida. Largura/altura devem ser > 0.")
    return Roi(x=x, y=y, w=w, h=h)


def _crop_roi(gray: np.ndarray, roi: Optional[Roi]) -> np.ndarray:
    if roi is None:
        return gray
    h, w = gray.shape[:2]
    x0 = max(0, min(roi.x, w - 1))
    y0 = max(0, min(roi.y, h - 1))
    x1 = max(x0 + 1, min(roi.x + roi.w, w))
    y1 = max(y0 + 1, min(roi.y + roi.h, h))
    return gray[y0:y1, x0:x1]


def _motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Score ~ [0..1] (aprox) baseado em energia de movimento.
    """
    diff = cv2.absdiff(gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)

    # normaliza (0..255) -> (0..1)
    score = float(np.mean(diff) / 255.0)
    return score


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    """
    Normaliza scores brutos para [0..1] via percentis robustos.
    Isso evita depender da escala absoluta da diferença de pixels.
    """
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float32)
    lo = float(np.percentile(arr, 10))
    hi = float(np.percentile(arr, 95))
    denom = max(hi - lo, 1e-6)
    normalized = np.clip((arr - lo) / denom, 0.0, 1.0)
    return [float(v) for v in normalized]


def _group_events(
    scores: Sequence[float],
    sample_frames: Sequence[int],
    *,
    threshold: float,
    max_gap: int,
) -> List[OcclusionEvent]:
    """
    scores indexado por sample, usando sample_frames para frame absoluto.
    """
    if len(scores) != len(sample_frames):
        raise ValueError("scores e sample_frames devem ter o mesmo tamanho.")

    events: List[OcclusionEvent] = []

    in_event = False
    start_i = 0
    peak = 0.0
    gap = 0

    for i, s in enumerate(scores):
        if s >= threshold:
            if not in_event:
                in_event = True
                start_i = i
                peak = s
                gap = 0
            else:
                gap = 0
                if s > peak:
                    peak = s
        else:
            if in_event:
                gap += 1
                if gap > max_gap:
                    end_i = i - gap
                    start_f = int(sample_frames[start_i])
                    end_f = int(sample_frames[end_i])
                    events.append(
                        OcclusionEvent(
                            start_frame=start_f,
                            end_frame=end_f,
                            score_peak=float(peak),
                            kind="occlusion",
                        )
                    )
                    in_event = False

    if in_event:
        end_i = len(scores) - 1
        start_f = int(sample_frames[start_i])
        end_f = int(sample_frames[end_i])
        events.append(
            OcclusionEvent(
                start_frame=start_f,
                end_frame=end_f,
                score_peak=float(peak),
                kind="occlusion",
            )
        )

    return events


def _collect_motion_scores(
    video_path: str,
    *,
    sample_every_n_frames: int,
    max_frames: Optional[int],
    roi: Optional[str],
) -> Tuple[List[int], List[float]]:
    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames deve ser >= 1.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    roi_spec = _parse_roi(roi)

    prev_full = _read_gray(cap)
    prev = _crop_roi(prev_full, roi_spec) if prev_full is not None else None
    if prev is None:
        cap.release()
        return [], []

    raw_scores: List[float] = []
    sample_frames: List[int] = []

    frame_abs = 1
    while True:
        gray_full = _read_gray(cap)
        if gray_full is None:
            break
        gray = _crop_roi(gray_full, roi_spec)

        if max_frames is not None and frame_abs >= max_frames:
            break

        if (frame_abs % sample_every_n_frames) == 0:
            raw_scores.append(_motion_score(prev, gray))
            sample_frames.append(frame_abs)

        prev = gray
        frame_abs += 1

    cap.release()
    return sample_frames, raw_scores


def detect_occlusions(
    video_path: str,
    *,
    sample_every_n_frames: int = 2,
    occlusion_threshold: float = 0.62,
    max_occlusion_gap: int = 6,
    max_frames: Optional[int] = None,
    roi: Optional[str] = None,
) -> List[OcclusionEvent]:
    """
    Retorna lista de eventos de oclusão (start/end em frames).

    - sample_every_n_frames: amostragem (maior => mais rápido, menos sensível)
    - occlusion_threshold: limiar do score (0..1)
    - max_occlusion_gap: número de samples abaixo do threshold ainda dentro do mesmo evento
    - max_frames: limita processamento para debug
    """
    sample_frames, raw_scores = _collect_motion_scores(
        video_path,
        sample_every_n_frames=sample_every_n_frames,
        max_frames=max_frames,
        roi=roi,
    )

    scores = _normalize_scores(raw_scores)
    adaptive_threshold = float(np.clip(occlusion_threshold, 0.0, 1.0))

    events = _group_events(
        scores,
        sample_frames,
        threshold=adaptive_threshold,
        max_gap=max_occlusion_gap,
    )
    return events


def detect_occlusions_debug(
    video_path: str,
    *,
    sample_every_n_frames: int = 2,
    occlusion_threshold: float = 0.62,
    max_occlusion_gap: int = 6,
    max_frames: Optional[int] = None,
    roi: Optional[str] = None,
) -> Tuple[List[OcclusionEvent], List[Dict[str, float | int]]]:
    """
    Versão de debug: retorna (eventos, amostras score por frame amostrado).
    """
    sample_frames, raw_scores = _collect_motion_scores(
        video_path,
        sample_every_n_frames=sample_every_n_frames,
        max_frames=max_frames,
        roi=roi,
    )
    samples: List[Dict[str, float | int]] = []

    scores = _normalize_scores(raw_scores)
    adaptive_threshold = float(np.clip(occlusion_threshold, 0.0, 1.0))

    events = _group_events(
        scores,
        sample_frames,
        threshold=adaptive_threshold,
        max_gap=max_occlusion_gap,
    )
    for frame, raw, normalized in zip(sample_frames, raw_scores, scores):
        samples.append(
            {
                "frame": int(frame),
                "raw_score": float(raw),
                "score": float(normalized),
                "threshold_used": adaptive_threshold,
            }
        )
    return events, samples
