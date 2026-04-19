"""
mocha_config_generator.py

Gera e valida um JSON de configuração (project_config.json) a partir:
- templates (config_templates.json)
- overrides de CLI

A ideia é ter um schema simples e previsível.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


class ConfigError(ValueError):
    pass


@dataclass
class ProjectConfig:
    profile: str
    fps: Optional[float]
    min_block_len: int
    safety_buffer: int
    sample_every_n_frames: int
    occlusion_threshold: float
    max_occlusion_gap: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "fps": self.fps,
            "min_block_len": self.min_block_len,
            "safety_buffer": self.safety_buffer,
            "sample_every_n_frames": self.sample_every_n_frames,
            "occlusion_threshold": self.occlusion_threshold,
            "max_occlusion_gap": self.max_occlusion_gap,
        }


def load_templates(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "profiles" not in data:
        raise ConfigError("config_templates.json inválido: esperado objeto com chave 'profiles'.")
    if not isinstance(data["profiles"], dict):
        raise ConfigError("config_templates.json inválido: 'profiles' deve ser um objeto.")
    return data["profiles"]


def build_config_from_profile(
    templates: Dict[str, Dict[str, Any]],
    profile: str,
    *,
    fps: Optional[float] = None,
    min_block_len: Optional[int] = None,
    safety_buffer: Optional[int] = None,
    sample_every_n_frames: Optional[int] = None,
    occlusion_threshold: Optional[float] = None,
    max_occlusion_gap: Optional[int] = None,
) -> ProjectConfig:
    if profile not in templates:
        raise ConfigError(f"Perfil '{profile}' não existe em config_templates.json.")

    base = dict(templates[profile])

    # defaults se não existirem no template
    base.setdefault("min_block_len", 12)
    base.setdefault("safety_buffer", 7)
    base.setdefault("sample_every_n_frames", 2)
    base.setdefault("occlusion_threshold", 0.62)
    base.setdefault("max_occlusion_gap", 6)
    base.setdefault("fps", None)

    # overrides
    if fps is not None:
        base["fps"] = fps
    if min_block_len is not None:
        base["min_block_len"] = int(min_block_len)
    if safety_buffer is not None:
        base["safety_buffer"] = int(safety_buffer)
    if sample_every_n_frames is not None:
        base["sample_every_n_frames"] = int(sample_every_n_frames)
    if occlusion_threshold is not None:
        base["occlusion_threshold"] = float(occlusion_threshold)
    if max_occlusion_gap is not None:
        base["max_occlusion_gap"] = int(max_occlusion_gap)

    # validações
    if base["min_block_len"] < 1:
        raise ConfigError("min_block_len deve ser >= 1.")
    if base["safety_buffer"] < 0:
        raise ConfigError("safety_buffer deve ser >= 0.")
    if base["sample_every_n_frames"] < 1:
        raise ConfigError("sample_every_n_frames deve ser >= 1.")
    if not (0.0 < base["occlusion_threshold"] < 1.0):
        raise ConfigError("occlusion_threshold deve estar entre 0 e 1.")
    if base["max_occlusion_gap"] < 0:
        raise ConfigError("max_occlusion_gap deve ser >= 0.")

    return ProjectConfig(
        profile=profile,
        fps=base["fps"],
        min_block_len=base["min_block_len"],
        safety_buffer=base["safety_buffer"],
        sample_every_n_frames=base["sample_every_n_frames"],
        occlusion_threshold=base["occlusion_threshold"],
        max_occlusion_gap=base["max_occlusion_gap"],
    )


def write_config(path: str, cfg: ProjectConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
        f.write("\n")