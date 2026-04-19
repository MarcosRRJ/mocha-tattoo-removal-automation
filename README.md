# Mocha Tattoo Removal Automation (Mocha Pro 2026)

Automação profissional (Python 3.8+) para **analisar o vídeo**, **detectar oclusões (mão/braço)** e **dividir automaticamente em blocos de tracking** para workflow no **Mocha Pro 2026** (remoção de tatuagem com múltiplas oclusões).

> Este projeto **não depende do Mocha** para rodar a análise (OpenCV). O resultado é exportado como **JSON + CSV + relatório** e um **script Python** (template) para importar/replicar a estrutura dentro do Mocha.

## Principais entregáveis

- `mocha_block_splitter.py` — CLI principal (análise + split em blocos + export)
- `occlusion_detector.py` — detecção de oclusões e “áreas limpas”
- `tracking_optimizer.py` — escolha/adaptação de parâmetros de tracking por bloco
- `mocha_config_generator.py` — geração/validação do JSON de configuração
- `mocha_script_exporter.py` — exporta um script Python (template) para o Mocha
- `config_templates.json` — perfis (pele clara/escura/tatuada)
- `example_workflow.py` — exemplo end-to-end

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido (CLI)

### 1) Analisar vídeo e gerar blocos

```bash
python mocha_block_splitter.py \
  --input "input.mov" \
  --out-dir "out" \
  --profile "skin_tattooed_medium" \
  --roi "520,260,280,220" \
  --min-block-len 12 \
  --safety-buffer 7 \
  --write-mocha-script
```

Saídas geradas em `out/`:
- `project_config.json`
- `blocks.csv`
- `occlusions.csv`
- `occlusion_scores.csv` (score por frame amostrado + threshold usado)
- `report.json`
- `mocha_import_script.py`

> Dica: para tattoo na lombar, use `--roi x,y,w,h` para analisar só a área relevante e aumentar a sensibilidade a mão/braço.

### 2) Rodar o exemplo end-to-end

```bash
python example_workflow.py --input "input.mov" --out-dir "out" --profile "skin_tattooed_medium"
```

## Como funciona (resumo)

1. **Leitura do vídeo** (OpenCV) e amostragem de frames.
2. **Score de oclusão** usando mudança de pixels/movimento (diferença temporal + máscara adaptativa).
3. **Agrupamento** de picos de oclusão em janelas (início/fim).
4. **Cálculo de blocos “limpos”** entre oclusões com `safety_buffer`.
5. **Otimização de parâmetros** por bloco (baseado em textura, movimento e perfil de pele).
6. **Export** (JSON/CSV) + **template de script** para recriar camadas no Mocha.

## Observações sobre o Mocha Pro 2026

- O Mocha possui API/scripting que pode variar por build. Aqui exportamos um **template** robusto, e os nomes de funções podem ser ajustados conforme a instalação.
- O pipeline foi feito para ser **modular**: você pode substituir a detecção por um modelo (ex. segmentação de mão) futuramente.

## Roadmap

- suporte a máscara por modelo (MediaPipe/YOLO) opcional
- visualização timeline em PNG
- integração com Nuke/After Effects via export adicional

## Licença

MIT
