"""
mocha_block_splitter.py

CLI principal:
- lê vídeo
- detecta oclusões
- gera blocos (ranges) para tracking
- exporta CSV/JSON + script template para Mocha

Nota: este módulo expõe analyze_and_export() para ser usado pelo example_workflow.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from mocha_script_exporter import MochaLayerSpec, build_mocha_python_script
from occlusion_detector import OcclusionEvent, detect_occlusions
from tracking_optimizer import choose_tracking_parameters


@dataclass(frozen=True)
class Block:
    index: int
    start_frame: int
    end_frame: int
    reason: str  # e.g. "between_occlusions"


def _video_fps_and_frames(video_path: str) -> Tuple[Optional[float], int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        fps_val: Optional[float] = float(fps)
    else:
        fps_val = None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps_val, frame_count


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _compute_blocks_from_occlusions(
    total_frames: int,
    occlusions: List[OcclusionEvent],
    *,
    min_block_len: int,
    safety_buffer: int,
) -> List[Block]:
    """
    Constrói blocos "limpos" fora das regiões de oclusão, adicionando safety_buffer
    ao redor de cada oclusão para reduzir risco de contaminação.
    """
    if total_frames <= 0:
        return []

    # cria janelas proibidas (occlusion + buffer)
    forbidden: List[Tuple[int, int]] = []
    for ev in occlusions:
        a = _clamp(ev.start_frame - safety_buffer, 0, total_frames - 1)
        b = _clamp(ev.end_frame + safety_buffer, 0, total_frames - 1)
        if b >= a:
            forbidden.append((a, b))

    forbidden.sort()

    # merge overlaps
    merged: List[Tuple[int, int]] = []
    for a, b in forbidden:
        if not merged or a > merged[-1][1] + 1:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    # segmentos limpos = complementos
    clean_segments: List[Tuple[int, int]] = []
    cursor = 0
    for a, b in merged:
        if cursor <= a - 1:
            clean_segments.append((cursor, a - 1))
        cursor = b + 1
    if cursor <= total_frames - 1:
        clean_segments.append((cursor, total_frames - 1))

    blocks: List[Block] = []
    idx = 0
    for s, e in clean_segments:
        if (e - s + 1) < min_block_len:
            continue
        blocks.append(Block(index=idx, start_frame=s, end_frame=e, reason="between_occlusions"))
        idx += 1

    # fallback: se tudo ficou pequeno, cria 1 bloco geral
    if not blocks and total_frames >= min_block_len:
        blocks = [Block(index=0, start_frame=0, end_frame=total_frames - 1, reason="fallback_full_range")]

    return blocks


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def analyze_and_export(
    *,
    input_path: str,
    out_dir: str,
    config: Dict[str, Any],
    write_mocha_script: bool,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fps, total_frames = _video_fps_and_frames(input_path)

    # config
    min_block_len = int(config.get("min_block_len", 12))
    safety_buffer = int(config.get("safety_buffer", 7))
    sample_every_n_frames = int(config.get("sample_every_n_frames", 2))
    occlusion_threshold = float(config.get("occlusion_threshold", 0.62))
    max_occlusion_gap = int(config.get("max_occlusion_gap", 6))

    # detect occlusions
    occlusions = detect_occlusions(
        input_path,
        sample_every_n_frames=sample_every_n_frames,
        occlusion_threshold=occlusion_threshold,
        max_occlusion_gap=max_occlusion_gap,
    )

    blocks = _compute_blocks_from_occlusions(
        total_frames=total_frames,
        occlusions=occlusions,
        min_block_len=min_block_len,
        safety_buffer=safety_buffer,
    )

    # export occlusions.csv
    occl_rows = [
        {
            "start_frame": ev.start_frame,
            "end_frame": ev.end_frame,
            "score_peak": f"{ev.score_peak:.4f}",
            "kind": ev.kind,
        }
        for ev in occlusions
    ]
    _write_csv(str(out / "occlusions.csv"), occl_rows, ["start_frame", "end_frame", "score_peak", "kind"])

    # export blocks.csv + layer specs
    block_rows: List[Dict[str, Any]] = []
    layer_specs: List[MochaLayerSpec] = []
    for b in blocks:
        params = choose_tracking_parameters(
            input_path,
            start_frame=b.start_frame,
            end_frame=b.end_frame,
            profile=str(config.get("profile", "default")),
        )
        block_rows.append(
            {
                "block_index": b.index,
                "start_frame": b.start_frame,
                "end_frame": b.end_frame,
                "reason": b.reason,
                "params": json.dumps(params, ensure_ascii=False),
            }
        )
        layer_specs.append(
            MochaLayerSpec(
                name=f"Track_Block_{b.index:03d}",
                start_frame=b.start_frame,
                end_frame=b.end_frame,
                parameters=params,
            )
        )
    _write_csv(str(out / "blocks.csv"), block_rows, ["block_index", "start_frame", "end_frame", "reason", "params"])

    # report.json
    report = {
        "input": os.path.abspath(input_path),
        "fps_detected": fps,
        "total_frames": total_frames,
        "config_used": config,
        "occlusion_count": len(occlusions),
        "block_count": len(blocks),
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # mocha script
    if write_mocha_script:
        script = build_mocha_python_script(
            layer_specs,
            project_fps=(float(config["fps"]) if config.get("fps") else fps),
            notes="Generated blocks for tracking; adjust Mocha API calls as needed.",
        )
        with open(out / "mocha_import_script.py", "w", encoding="utf-8") as f:
            f.write(script)


def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="mocha_block_splitter",
        description="Analisa vídeo, detecta oclusões e gera blocos de tracking para Mocha Pro.",
    )
    p.add_argument("--input", required=True, help="Caminho do vídeo (mov/mp4).")
    p.add_argument("--out-dir", required=True, help="Diretório de saída.")
    p.add_argument("--profile", default="skin_tattooed_medium", help="Perfil em config_templates.json.")
    p.add_argument("--templates", default="config_templates.json", help="Caminho do config_templates.json.")
    p.add_argument("--min-block-len", type=int, default=None, help="Override do tamanho mínimo do bloco.")
    p.add_argument("--safety-buffer", type=int, default=None, help="Override do buffer de segurança.")
    p.add_argument("--sample-every-n-frames", type=int, default=None, help="Amostragem do detector (N).")
    p.add_argument("--occlusion-threshold", type=float, default=None, help="Threshold do score (0-1).")
    p.add_argument("--max-occlusion-gap", type=int, default=None, help="Gap máx (frames amostrados) para unir eventos.")
    p.add_argument("--write-mocha-script", action="store_true", help="Gera mocha_import_script.py em out-dir.")
    args = p.parse_args()

    from mocha_config_generator import build_config_from_profile, load_templates, write_config

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    templates = load_templates(args.templates)
    cfg = build_config_from_profile(
        templates,
        args.profile,
        min_block_len=args.min_block_len,
        safety_buffer=args.safety_buffer,
        sample_every_n_frames=args.sample_every_n_frames,
        occlusion_threshold=args.occlusion_threshold,
        max_occlusion_gap=args.max_occlusion_gap,
    )

    # escreve config efetiva
    write_config(str(out / "project_config.json"), cfg)

    analyze_and_export(
        input_path=args.input,
        out_dir=str(out),
        config=cfg.to_dict(),
        write_mocha_script=bool(args.write_mocha_script),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())