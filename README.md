# Aprendizaje in silico de la optimización AV/VV en TRC a partir de la impedancia intracardíaca

[![Licencia: MIT](https://img.shields.io/badge/c%C3%B3digo-MIT-blue.svg)](LICENSE)
[![Manuscrito: CC BY 4.0](https://img.shields.io/badge/manuscrito-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0.txt)

Este repositorio contiene el código, la configuración, los artefactos de
resultados y el manuscrito de un estudio de simulación que evalúa si una señal
de **impedancia intracardíaca** junto con el contexto del dispositivo permite
elegir la programación AV/VV que maximiza el trabajo sistólico en **terapia de
resincronización cardíaca (TRC)**.

> **Estado:** estudio in silico de factibilidad. No es validación clínica. Todos
> los pacientes, señales y etiquetas son simulados.

- **Manuscrito:** [`manuscript/preprint_es.tex`](manuscript/preprint_es.tex) · [PDF](manuscript/preprint_es.pdf)
- **Código del pipeline:** [`phase0_crt_ml/`](phase0_crt_ml/)
- **Resultados primarios:** [`phase0_crt_ml/reports/`](phase0_crt_ml/reports/)
- **Guía de reproducción:** [`phase0_crt_ml/REPRODUCING.md`](phase0_crt_ml/REPRODUCING.md)

## Resumen

Se generaron 500 pacientes virtuales con el modelo cardiovascular **CircAdapt**
(VanOsta2024) y se barrió una grilla de **8 AV × 7 VV = 56 programaciones** por
paciente (28.000 observaciones etiquetadas; 3 latidos por programación, se
analiza el último en estado estacionario). Un modelo
fenomenológico de observación agregó ganancia, offset, ruido y deriva de línea
base a la impedancia. Se extrajeron **42 features interpretables** y el problema
se formuló como *learning-to-rank*: cada paciente es una consulta y sus 56
programaciones son los documentos a ordenar.

Se compararon un ranker lineal por pares, LambdaMART, una CNN 1D pequeña,
variantes híbridas con embeddings de la CNN y un *blend* ponderado por
validación. La métrica primaria es la fracción de pacientes cuyo ajuste elegido
queda dentro del **2 %** del mejor trabajo sistólico; el **5 %** es el endpoint
pragmático preespecificado. El dominio de impedancia desplazado es un conjunto
de **validación externa sellado**: nunca se usó para entrenar ni para elegir
hiperparámetros.

## Estructura del repositorio

```text
heart-vanosta/
├── manuscript/
│   ├── preprint_es.tex          # manuscrito (fuente LaTeX)
│   ├── preprint_es.pdf          # PDF compilado
│   ├── references.bib
│   └── figures/                 # figuras generadas desde los reportes
└── phase0_crt_ml/
    ├── configs/default.yaml     # grilla AV/VV, rangos de pacientes, hiperparámetros
    ├── src/phase0/
    │   ├── benchmark.py         # dataset de benchmark
    │   ├── dataset.py           # dataset del experimento inicial
    │   ├── evaluate.py          # métricas, splits, ablaciones
    │   ├── features.py          # 42 features de impedancia + contexto
    │   ├── models/              # ranker lineal, LambdaMART, CNN
    │   ├── physiology/          # adaptador CircAdapt
    │   └── sensors/             # modelo de observación de impedancia
    ├── scripts/                 # pipeline, ablaciones, figuras y utilidades
    ├── tests/                   # 22 tests
    ├── reports/                 # artefactos curados de la corrida primaria
    ├── requirements.txt
    ├── README.md                # documentación detallada en español
    └── REPRODUCING.md           # entorno, comandos y salidas esperadas
```

## Requisitos

- **Python 3.12** (versión usada en la corrida reportada).
- **CircAdapt 2602** (`pyCircAdapt`), distribuido como wheel por el equipo de
  CircAdapt. No está en PyPI; el script `scripts/setup_circadapt.sh` lo instala
  para Linux x86_64, macOS universal2 y Windows x86_64.
- `numpy`, `scipy`, `matplotlib`, `pyyaml`, `lightgbm`, `torch` y `pytest`
  (ver `requirements.txt`).

## Instalación rápida

```bash
git clone https://github.com/ForTheLoveOfPointers/heart-vanosta.git
cd heart-vanosta/phase0_crt_ml

/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv   # macOS con Homebrew
# python3.12 -m venv .venv                                   # Linux
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt pytest torch lightgbm
bash scripts/setup_circadapt.sh
.venv/bin/python scripts/smoke_test.py
.venv/bin/python -m pytest -q
```

Salida esperada del smoke test:

```text
SMOKE OK
samples=36 features=42
```

### Nota para macOS: torch y LightGBM

En macOS, las wheels de `torch` y `lightgbm` traen dos copias de `libomp`. Si se
cargan en el mismo proceso, el programa aborta con
`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`.
Para evitarlo, compilá LightGBM sin OpenMP (los experimentos son lo bastante
pequeños como para no necesitarlo):

```bash
brew install libomp
.venv/bin/pip install --no-binary lightgbm \
  --config-settings=cmake.define.USE_OPENMP=OFF "lightgbm==4.7.0"
```

En Linux, la instalación estándar `pip install lightgbm` funciona sin problemas.

## Reproducción completa

La corrida primaria se compone de cinco pasos. Tarda aproximadamente una hora en
un Apple Silicon de 8 núcleos y genera ~25 MB en `reports/` (los `.npz`
grandes no se versionan; ver `REPRODUCING.md`). También podés correr todo con un
solo comando:

```bash
cd phase0_crt_ml
bash scripts/reproduce_all.sh        # THREADS=10 para más hilos
```

Equivalentemente, paso a paso:

```bash
cd phase0_crt_ml

# 1. Dataset de 500 pacientes + benchmark lineal y CNN
.venv/bin/python scripts/run_benchmark.py \
  --patients 500 --signal-length 256 --tag bench500_imp --threads 8

# 2. Escalera de modelos (sin normalización within-patient)
.venv/bin/python scripts/run_gbt_benchmark.py \
  --dataset-tag bench500_imp --out-tag ladder500_imp_reconcile \
  --patient-normalize none --threads 8

# 3. Control negativo con etiquetas barajadas intra-paciente
.venv/bin/python scripts/negative_control.py --tag bench500_imp --seed 0

# 4. Curva de aprendizaje
.venv/bin/python scripts/benchmark_learning_curve.py \
  --tag bench500_imp --threads 8

# 5. Reconciliación del protocolo de dominio desplazado (2 % y 5 %)
.venv/bin/python scripts/reconcile_shift_protocol.py

# 6. Figuras del manuscrito (se regeneran desde los reportes)
.venv/bin/python scripts/make_manuscript_figures.py
```

Los artefactos curados que sí se versionan son los `.json`, `.md`, `.png` y
`scores_*.npz` de `reports/`. Los datasets y checkpoints grandes se regeneran
con el pipeline.

## Resultados principales

Baseline primario sin normalización within-patient (split 350/75/75, semilla 0).
Umbral estricto del 2 %; regret = pérdida relativa media de trabajo sistólico.

| Modelo | Casi óptimo source | Regret source | Casi óptimo shifted | Regret shifted |
|---|---:|---:|---:|---:|
| Linear (impedancia + contexto) | 0.640 | 0.0180 | 0.640 | 0.0217 |
| Linear sin sensores de onda | 0.493 | 0.0366 | 0.493 | 0.0366 |
| LambdaMART sobre features | 0.680 | 0.0248 | 0.600 | 0.0317 |
| Tiny CNN sobre señal cruda | 0.107 | 0.1561 | 0.107 | 0.1467 |
| LambdaMART sobre embeddings CNN | 0.667 | 0.0240 | 0.507 | 0.0361 |
| LambdaMART híbrido (features + embeddings) | 0.693 | 0.0236 | 0.587 | 0.0248 |
| Linear sobre features + embeddings CNN | **0.800** | **0.0109** | **0.680** | **0.0205** |
| Blend ponderado por validación | **0.800** | **0.0109** | **0.680** | **0.0205** |

Al 5 % de tolerancia, el modelo lineal pasa de 0.880 a 0.867 entre dominios; el
linear+embeddings, de 0.947 a 0.893; el blend es idéntico a este último; el piso
sin sensores es 0.680 en ambos dominios. El control negativo colapsa a 0.080 de
casi óptimo (Spearman −0.149), lo que descarta una fuga de información obvia.

## Qué prueba y qué no prueba

**Prueba** que, en este simulador, la impedancia contiene información suficiente
para ordenar programaciones AV/VV mejor que el azar y mejor que el contexto
solo, y que un ranker lineal simple conserva gran parte del rendimiento bajo un
desplazamiento extremo del sensor.

**No prueba** que la impedancia real permita elegir AV/VV con precisión clínica.
El modelo de impedancia es fenomenológico y no captura posición del electrodo,
hematocrito, postura, respiración ni fibrosis. El óptimo VV está influido por
una penalización de resincronización. La validación requiere un estudio
prospectivo con impedancia real y una referencia hemodinámica independiente.

Más limitaciones en el manuscrito y en `phase0_crt_ml/README.md`.

## Cómo citar

Si usás el código o los resultados, citá el repositorio y el preprint. Los
metadatos están en [`CITATION.cff`](CITATION.cff).

```bibtex
@software{costa_heart_vanosta_2026,
  author  = {Costa, Lucas},
  title   = {Aprendizaje in silico de la optimización AV/VV en TRC a partir de la impedancia intracardíaca},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/ForTheLoveOfPointers/heart-vanosta},
  license = {MIT}
}
```

## Licencias

- **Código** (`phase0_crt_ml/` y scripts): [MIT](LICENSE).
- **Manuscrito, figuras y texto** (`manuscript/`): [CC BY 4.0](LICENSE-CC-BY-4.0.txt).

## Contacto

Lucas Costa — `2000.costalucas@gmail.com`

Las contribuciones, correcciones y *issues* son bienvenidos.
