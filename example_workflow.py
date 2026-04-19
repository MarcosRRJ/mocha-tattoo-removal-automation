"""
example_workflow.py

Exemplo end-to-end:
- carrega templates
- monta config
- roda análise (block_splitter)
- exporta JSON/CSV + script template

Execute:
python example_workflow.py --input input.mov --out-dir out --profile skin_tattooed_medium
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mocha_config_generator import build_config_from_profile, load_templates, write_config
from mocha_block_splitter import analyze_and_export


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Vídeo de entrada (mov/mp4).")
    p.add_argument("--out-dir", required=True, help="Diretório de saída.")
    p.add_argument("--profile", default="skin_tattooed_medium", help="Perfil do config_templates.json.")
    p.add_argument("--templates", default="config_templates.json", help="Caminho do config_templates.json.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = load_templates(args.templates)
    cfg = build_config_from_profile(templates, args.profile)

    cfg_path = out_dir / "project_config.json"
    write_config(str(cfg_path), cfg)

    analyze_and_export(
        input_path=args.input,
        out_dir=str(out_dir),
        config=cfg.to_dict(),
        write_mocha_script=True,
    )


if __name__ == "__main__":
    main()