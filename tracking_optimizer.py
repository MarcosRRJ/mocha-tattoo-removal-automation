"""
tracking_optimizer.py

Otimizador adaptativo (heurístico) de parâmetros de tracking para cada bloco.
Sem acesso direto ao Mocha, inferimos parâmetros com base em:
- textura (variância do frame)
- movimento (diferença temporal)
- perfil (pele clara/escura/tatuada)
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def _sample_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Não foi possível ler frame {frame_index} de {video_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def _texture_score(gray: np.ndarray) -> float:
    # variância normalizada
    v = float(np.var(gray) / (255.0 * 255.0))
    return max(0.0, min(1.0, v * 6.0))


def _motion_score(a: np.ndarray, b: np.ndarray) -> float:
    diff = cv2.absdiff(a, b)
    return float(np.mean(diff) / 255.0)


def choose_tracking_parameters(
    video_path: str,
    *,
    start_frame: int,
    end_frame: int,
    profile: str,
) -> Dict[str, Any]:
    """
    Retorna dicionário com parâmetros sugeridos para o Mocha.

    Você pode mapear esses campos para o que sua versão do Mocha aceitar.
    """
    mid = int((start_frame + end_frame) / 2)
    a = _sample_frame(video_path, max(0, mid - 2))
    b = _sample_frame(video_path, max(0, mid + 2))

    tex = _texture_score(a)
    mot = _motion_score(a, b)

    # base defaults
    # (nomes genéricos; ajuste depois no exporter)
    params: Dict[str, Any] = {
        "search_radius": 20,
        "min_feature_size": 8,
        "motion_model": "perspective",  # could be: translation/affine/perspective
        "use_contrast_enhancement": True,
        "texture_score": round(tex, 4),
        "motion_score": round(mot, 4),
        "profile": profile,
    }

    # perfil
    if "tattoo" in profile:
        params["min_feature_size"] = 6
        params["search_radius"] = 26
    if "dark" in profile:
        params["use_contrast_enhancement"] = True
        params["search_radius"] = max(params["search_radius"], 24)

    # adaptação por movimento
    if mot > 0.20:
        params["search_radius"] = max(params["search_radius"], 32)
        params["motion_model"] = "perspective"
    elif mot > 0.10:
        params["search_radius"] = max(params["search_radius"], 26)
        params["motion_model"] = "affine"
    else:
        params["motion_model"] = "affine" if tex < 0.15 else "perspective"

    # adaptação por textura
    if tex < 0.10:
        params["min_feature_size"] = max(params["min_feature_size"], 10)
        params["use_contrast_enhancement"] = True

    return params