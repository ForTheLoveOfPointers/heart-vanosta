# Aprendizaje in silico de la optimización AV/VV en TRC a partir de la impedancia intracardíaca

[![Licencia: MIT](https://img.shields.io/badge/c%C3%B3digo-MIT-blue.svg)](LICENSE)
[![Manuscrito: CC BY 4.0](https://img.shields.io/badge/manuscrito-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24487582.svg)](https://doi.org/10.5281/zenodo.24487582)

Este repositorio contiene el código, la configuración, los artefactos de
resultados y el manuscrito de un estudio de simulación que evalúa si una señal
de **impedancia intracardíaca** junto con el contexto del dispositivo permite
elegir la programación AV/VV que maximiza el trabajo sistólico en **terapia de
resincronización cardíaca (TRC)**.

> **Estado:** estudio in silico de factibilidad. No constituye validación
> clínica. Todos los pacientes, las señales y las etiquetas son simulados.

- **Manuscrito:** [`manuscript/preprint_es.tex`](manuscript/preprint_es.tex) · [PDF](manuscript/preprint_es.pdf)
- **Código del flujo de trabajo:** [`phase0_crt_ml/`](phase0_crt_ml/)
- **Resultados principales:** [`phase0_crt_ml/reports/`](phase0_crt_ml/reports/)
- **Guía de reproducción:** [`phase0_crt_ml/REPRODUCING.md`](phase0_crt_ml/REPRODUCING.md)

## Resumen

Se generaron 500 pacientes virtuales con el modelo cardiovascular **CircAdapt**
(VanOsta2024) y se recorrió una rejilla de **8 AV × 7 VV = 56 programaciones** por
paciente (28 000 observaciones etiquetadas; 3 latidos por programación, de los
cuales se analiza el último, asumido en estado estacionario). Un modelo
fenomenológico de observación añadió ganancia, desplazamiento, ruido y deriva de
línea base a la impedancia; estos parámetros parásitos se regeneran en cada
latido, lo que representa un caso deliberadamente adverso. Se extrajeron **42
características interpretables** y el problema se formuló como aprendizaje para
ordenar (*learning-to-rank*): cada paciente es una consulta y sus 56
programaciones son los documentos que deben ordenarse.

Se compararon un modelo de ordenación lineal por pares, LambdaMART, una CNN
unidimensional pequeña, variantes híbridas con embeddings de la CNN y una mezcla
ponderada por validación. La métrica principal es la fracción de pacientes cuyo
ajuste elegido queda dentro del **2 %** del mejor trabajo sistólico; el **5 %** es
el criterio pragmático preespecificado. El dominio de impedancia desplazado es un
conjunto de **prueba externa sellado**: nunca se usó para entrenar ni para elegir
hiperparámetros.

## Estructura del repositorio

```text
heart-vanosta/
├── manuscript/
│   ├── preprint_es.tex          # manuscrito (fuente LaTeX)
│   ├── preprint_es.pdf          # PDF compilado
│   ├── references.bib
│   └── figures/                 # figuras generadas a partir de los reportes
└── phase0_crt_ml/
    ├── configs/default.yaml     # rejilla AV/VV, rangos de pacientes, hiperparámetros
    ├── src/phase0/
    │   ├── benchmark.py         # conjunto de datos del banco de pruebas
    │   ├── evaluate.py          # métricas y particiones
    │   ├── features.py          # 42 características de impedancia + contexto
    │   ├── models/              # ordenación lineal, LambdaMART, CNN
    │   ├── physiology/          # adaptador CircAdapt
    │   └── sensors/             # modelo de observación de impedancia
    ├── scripts/                 # flujo de trabajo, figuras y utilidades
    ├── tests/                   # 23 pruebas
    ├── reports/                 # artefactos curados de la ejecución principal
    ├── requirements.txt
    ├── README.md                # documentación detallada en español
    └── REPRODUCING.md           # entorno, comandos y salidas esperadas
```

## Requisitos

- **Python 3.12** (versión usada en la ejecución reportada).
- **CircAdapt 2602** (`pyCircAdapt`), distribuido como *wheel* por el equipo de
  CircAdapt. No está en PyPI; el script `scripts/setup_circadapt.sh` lo instala
  para Linux x86_64, macOS universal2 y Windows x86_64.
- `numpy`, `scipy`, `matplotlib`, `pyyaml`, `lightgbm`, `torch` y `pytest`
  (véase `requirements.txt`).

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

Salida esperada de la prueba mínima:

```text
SMOKE OK
samples=27 features=42
```

### Nota para macOS: torch y LightGBM

En macOS, los *wheels* de `torch` y `lightgbm` incluyen dos copias de `libomp`.
Si se cargan en el mismo proceso, el programa aborta con
`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`.
Para evitarlo, compile LightGBM sin OpenMP (los experimentos son lo bastante
pequeños como para no necesitarlo):

```bash
brew install libomp
.venv/bin/pip install --no-binary lightgbm \
  --config-settings=cmake.define.USE_OPENMP=OFF "lightgbm==4.7.0"
```

En Linux, la instalación estándar `pip install lightgbm` funciona sin problemas.

## Reproducción completa

La ejecución principal consta de cinco pasos. Tarda aproximadamente una hora en
un equipo Apple Silicon de 8 núcleos y genera ~25 MB en `reports/` (los archivos
`.npz` grandes no se versionan; véase `REPRODUCING.md`). También puede ejecutar
todo con un solo comando:

```bash
cd phase0_crt_ml
bash scripts/reproduce_all.sh        # THREADS=10 para más hilos
```

De forma equivalente, paso a paso:

```bash
cd phase0_crt_ml

# 1. Conjunto de datos de 500 pacientes + línea base lineal y CNN
.venv/bin/python scripts/run_benchmark.py \
  --patients 500 --signal-length 256 --tag bench500_imp --threads 8

# 2. Escalera de modelos (sin normalización intrapaciente; incluye el control
#    negativo con etiquetas barajadas dentro de cada paciente)
.venv/bin/python scripts/run_gbt_benchmark.py \
  --dataset-tag bench500_imp --out-tag ladder500_imp_reconcile \
  --patient-normalize none --threads 8

# 3. Curva de aprendizaje
.venv/bin/python scripts/benchmark_learning_curve.py \
  --tag bench500_imp --threads 8

# 4. Reconciliación del protocolo de dominio desplazado (2 % y 5 %)
.venv/bin/python scripts/reconcile_shift_protocol.py

# 5. Figuras del manuscrito (se regeneran a partir de los reportes)
.venv/bin/python scripts/make_manuscript_figures.py
```

Los artefactos curados que se versionan son los archivos `.json`, `.md`, `.png`
y `scores_*.npz` de `reports/`. Los conjuntos de datos y los *checkpoints*
grandes se regeneran con el flujo de trabajo.

## Resultados principales

Línea base principal sin normalización intrapaciente (partición 350/75/75,
semilla 0). Umbral estricto del 2 %; la pérdida es la pérdida relativa media de
trabajo sistólico.

| Modelo | Casi óptimo (origen) | Pérdida (origen) | Casi óptimo (desplazado) | Pérdida (desplazado) |
|---|---:|---:|---:|---:|
| Lineal (impedancia + contexto) | 0.640 | 0.0180 | 0.640 | 0.0217 |
| Lineal sin la onda de impedancia | 0.493 | 0.0366 | 0.493 | 0.0366 |
| LambdaMART sobre características | 0.680 | 0.0248 | 0.600 | 0.0317 |
| CNN pequeña sobre la señal cruda | 0.107 | 0.1561 | 0.107 | 0.1467 |
| LambdaMART sobre embeddings de la CNN | 0.667 | 0.0240 | 0.507 | 0.0361 |
| LambdaMART híbrido (características + embeddings) | 0.693 | 0.0236 | 0.587 | 0.0248 |
| Lineal sobre características + embeddings de la CNN | **0.800** | **0.0109** | **0.680** | **0.0205** |
| Mezcla ponderada por validación | **0.800** | **0.0109** | **0.680** | **0.0205** |

Con una tolerancia del 5 %, el modelo lineal pasa de 0.880 a 0.867 entre dominios;
el modelo lineal con embeddings, de 0.947 a 0.893; la mezcla es idéntica a este
último; el suelo sin la onda de impedancia es 0.680 en ambos dominios. El control
negativo se reduce a 0.080 de casi óptimo (Spearman −0.149), lo que descarta una
fuga de información evidente.

## Qué prueba y qué no prueba

**Prueba** que, en este simulador, la impedancia contiene información suficiente
para ordenar programaciones AV/VV mejor que el azar y mejor que el modelo sin
onda (contexto y programación), y que un modelo de ordenación lineal simple
conserva gran parte del rendimiento bajo un desplazamiento extremo del sensor.

**No prueba** que la impedancia real permita elegir AV/VV con precisión clínica.
El modelo de impedancia es fenomenológico y no captura la posición del electrodo,
el hematocrito, la postura, la respiración ni la fibrosis. El óptimo VV está
influido por una penalización de resincronización. Las características de contexto
(AV intrínseco, QRS y frecuencia cardíaca) son los valores verdaderos del
simulador y no estimaciones del dispositivo, de modo que el suelo sin la onda de
impedancia es optimista. La validación requiere un estudio prospectivo con
impedancia real y una referencia hemodinámica independiente.

En el manuscrito y en `phase0_crt_ml/README.md` se detallan las limitaciones
restantes.

## Cómo citar

Si utiliza el código o los resultados, cite el repositorio y el preprint. Los
metadatos están en [`CITATION.cff`](CITATION.cff).

```bibtex
@software{costa_heart_vanosta_2026,
  author  = {Costa, Lucas},
  title   = {Aprendizaje in silico de la optimización AV/VV en TRC a partir de la impedancia intracardíaca},
  year    = {2026},
  version = {1.1.1},
  doi     = {10.5281/zenodo.24487582},
  url     = {https://github.com/ForTheLoveOfPointers/heart-vanosta},
  license = {MIT}
}
```

La versión `v1.1.1` está archivada en Zenodo bajo el *DOI conceptual*
[`10.5281/zenodo.24487582`](https://doi.org/10.5281/zenodo.24487582), que remite
siempre a la última versión. El DOI específico de `v1.1.1` se anuncia en la
versión publicada en GitHub.

## Licencias

- **Código** (`phase0_crt_ml/` y scripts): [MIT](LICENSE).
- **Manuscrito, figuras y texto** (`manuscript/`): [CC BY 4.0](LICENSE-CC-BY-4.0.txt).

## Contacto

Lucas Costa — `2000.costalucas@gmail.com`

Las contribuciones, correcciones y *issues* son bienvenidos.
