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
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class OcclusionEvent:
    start_frame: int
    end_frame: int
    score_peak: float
    kind: str = "occlusion"


def _read_gray(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def _motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Score ~ [0..1] (aprox) baseado em energia de movimento.
    """
    diff = cv2.absdiff(gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)

    # normaliza (0..255) -> (0..1)
    score = float(np.mean(diff) / 255.0)
    return score


def _group_events(
    scores: List[float],
    *,
    sample_every_n_frames: int,
    threshold: float,
    max_gap: int,
) -> List[OcclusionEvent]:
    """
    scores indexado por 'sample index' (não por frame absoluto).
    Converte para ranges em frame absoluto: idx * sample_every_n_frames.
    """
    events: List[OcclusionEvent] = []

    in_event = False
    start_i = 0
    peak = 0.0
    peak_i = 0
    gap = 0

    for i, s in enumerate(scores):
        if s >= threshold:
            if not in_event:
                in_event = True
                start_i = i
                peak = s
                peak_i = i
                gap = 0
            else:
                gap = 0
                if s > peak:
                    peak = s
                    peak_i = i
        else:
            if in_event:
                gap += 1
                if gap > max_gap:
                    end_i = i - gap
                    start_f = start_i * sample_every_n_frames
                    end_f = end_i * sample_every_n_frames
                    events.append(
                        OcclusionEvent(
                            start_frame=int(start_f),
                            end_frame=int(end_f),
                            score_peak=float(peak),
                            kind="occlusion",
                        )
                    )
                    in_event = False

    if in_event:
        end_i = len(scores) - 1
        start_f = start_i * sample_every_n_frames
        end_f = end_i * sample_every_n_frames
        events.append(
            OcclusionEvent(
                start_frame=int(start_f),
                end_frame=int(end_f),
                score_peak=float(peak),
                kind="occlusion",
            )
        )

    return events


def detect_occlusions(
    video_path: str,
    *,
    sample_every_n_frames: int = 2,
    occlusion_threshold: float = 0.62,
    max_occlusion_gap: int = 6,
    max_frames: Optional[int] = None,
) -> List[OcclusionEvent]:
    """
    Retorna lista de eventos de oclusão (start/end em frames).

    - sample_every_n_frames: amostragem (maior => mais rápido, menos sensível)
    - occlusion_threshold: limiar do score (0..1)
    - max_occlusion_gap: número de samples abaixo do threshold ainda dentro do mesmo evento
    - max_frames: limita processamento para debug
    """
    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames deve ser >= 1.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    prev = _read_gray(cap)
    if prev is None:
        cap.release()
        return []

    scores: List[float] = []

    # frame absoluto atual no arquivo
    frame_abs = 1
    sample_idx = 0

    while True:
        gray = _read_gray(cap)
        if gray is None:
            break

        if max_frames is not None and frame_abs >= max_frames:
            break

        if (frame_abs % sample_every_n_frames) == 0:
            s = _motion_score(prev, gray)
            scores.append(s)
            sample_idx += 1

        prev = gray
        frame_abs += 1

    cap.release()

    # threshold robusto: se vídeo for muito estável, baixa o threshold efetivo
    if scores:
        p90 = float(np.percentile(np.array(scores, dtype=np.float32), 90))
        # garante que não fica impossível detectar em vídeos muito “calmos”
        adaptive_threshold = max(occlusion_threshold, min(0.85, p90 * 0.85))
    else:
        adaptive_threshold = occlusion_threshold

    return _group_events(
        scores,
        sample_every_n_frames=sample_every_n_frames,
        threshold=adaptive_threshold,
        max_gap=max_occlusion_gap,
    )