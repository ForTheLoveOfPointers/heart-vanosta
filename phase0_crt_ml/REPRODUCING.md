# Reproducción del estudio Phase 0 in silico

Este documento es la contraparte de reproducibilidad de
[`../manuscript/preprint_es.tex`](../manuscript/preprint_es.tex). Registra el
entorno exacto, los comandos, las semillas y los artefactos esperados de la
corrida primaria.

## 1. Entorno verificado

Corrida usada para regenerar los resultados del manuscrito:

- **Sistema:** macOS 15.6 (Darwin, arm64), Apple M2, 8 núcleos, 24 GB RAM.
- **Python:** 3.12.16 (Homebrew).
- **pyCircAdapt:** 2602 (wheel `circadapt-2602-py3-none-macosx_12_0_universal2.whl`).
- **numpy:** 2.5.5 · **scipy:** 1.18.1 · **matplotlib:** 3.11.2 ·
  **PyYAML:** 6.0.4.
- **torch:** 2.15.0.
- **lightgbm:** 4.7.0, compilado en macOS con `USE_OPENMP=OFF` (ver nota de
  macOS más abajo).
- **Semilla primaria:** `0` (config `seed: 0`; benchmark `--seed 0`).
- **Split:** 350 train / 75 validation / 75 test pacientes.
- **Tag del dataset:** `bench500_imp`. **Tag del ladder:** `ladder500_imp_reconcile`.

Para regenerar el registro del entorno:

```bash
python3.12 --version
sw_vers
uname -m
git rev-parse HEAD
.venv/bin/pip freeze > pip-freeze.txt
.venv/bin/python -c "import circadapt; print(getattr(circadapt, '__version__', '?'))"
```

El archivo `pip-freeze.txt` se versiona junto con el release. Los `.npz`
grandes y los checkpoints no se versionan: se regeneran con el pipeline.

## 2. Instalación

```bash
cd phase0_crt_ml
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt pytest torch lightgbm
bash scripts/setup_circadapt.sh
.venv/bin/python scripts/smoke_test.py
.venv/bin/python -m pytest -q
```

`setup_circadapt.sh` instala el wheel de CircAdapt desde el sitio del proyecto.
Se puede fijar la versión y verificar el checksum:

```bash
CIRCADAPT_VERSION=2602 CIRCADAPT_SHA256=<sha256-del-wheel> \
  bash scripts/setup_circadapt.sh
```

Resultado esperado del smoke test:

```text
SMOKE OK
samples=27 features=42
```

Y de la suite de tests:

```text
23 passed
```

### Nota para macOS: conflicto de `libomp` entre torch y LightGBM

Las wheels de `torch` y `lightgbm` para macOS incluyen cada una su propia copia
de OpenMP. Cargarlas en el mismo proceso aborta con:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Solución usada en la corrida reportada:

```bash
brew install libomp
.venv/bin/pip install --no-binary lightgbm \
  --config-settings=cmake.define.USE_OPENMP=OFF "lightgbm==4.7.0"
```

En Linux, `pip install lightgbm` funciona sin este paso.

## 3. Análisis primario

El baseline primario es la ruta **sin normalización within-patient**
(`patient_normalize: []`, `features.mode: raw`). El benchmark lineal/CNN crea
el dataset `bench500_imp` que reutilizan los pasos siguientes. **No** usar
`--skip-cnn`: la escalera de modelos necesita los embeddings de la CNN.

Para correr todo de una vez:

```bash
THREADS=8 bash scripts/reproduce_all.sh
```

Equivalente paso a paso:

```bash
# 1. Dataset de 500 pacientes + benchmark lineal y CNN
#    (incluye el dominio desplazado sellado)
.venv/bin/python scripts/run_benchmark.py \
  --patients 500 --signal-length 256 --tag bench500_imp --threads 8

# 2. Escalera: LambdaMART, embeddings CNN, híbridos, blend y control negativo
.venv/bin/python scripts/run_gbt_benchmark.py \
  --dataset-tag bench500_imp \
  --out-tag ladder500_imp_reconcile \
  --patient-normalize none --threads 8

# 3. Curva de aprendizaje
.venv/bin/python scripts/benchmark_learning_curve.py \
  --tag bench500_imp --threads 8

# 4. Reconciliación del protocolo sellado a 2 % y 5 %
.venv/bin/python scripts/reconcile_shift_protocol.py

# 5. Figuras del manuscrito
.venv/bin/python scripts/make_manuscript_figures.py
```

Tiempo aproximado: 1 hora en un Apple M2 de 8 núcleos. Espacio en disco:
~25 MB de `reports/`.

## 4. Salidas esperadas

Artefactos primarios del manuscrito:

| Artefacto | Origen |
|---|---|
| Dataset, splits y arrays source/shifted | `reports/benchmark_dataset_bench500_imp.npz`, `reports/benchmark_shifted_bench500_imp.npz`, `reports/benchmark_split_bench500_imp.json` |
| Resumen lineal/CNN | `reports/benchmark_summary_bench500_imp.md` |
| Escalera + ICs + blend (primario) | `reports/gbt_report_ladder500_imp_reconcile.json`, `reports/gbt_summary_ladder500_imp_reconcile.{md,png}` |
| Scores por modelo (auditoría) | `reports/scores_ladder500_imp_reconcile.npz` |
| Protocolo sellado a 2 % y 5 % | `reports/reconcile_shift_protocol.json` |
| Curva de aprendizaje | `reports/benchmark_learning_curve_bench500_imp.{json,png}` |
| Figuras del manuscrito | `manuscript/figures/{response_surface,model_comparison,learning_curve,ablations}.png` |

**Advertencia de procedencia.** `reports/pitch_summary_bench500_pp.md` es un
artefacto histórico de una corrida exploratoria y no forma parte de este release.
Las cifras primarias son las de la corrida `ladder500_imp_reconcile`;
`reports/benchmark_summary_bench500_imp.md` corresponde a la misma corrida
primaria sin normalización.

El control negativo citado en el manuscrito es el del ladder (GBT con etiquetas
barajadas: 0.080 y Spearman −0.149); se genera dentro de
`run_gbt_benchmark.py` y no requiere un paso aparte.

## 5. Compilación del manuscrito

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error preprint_es.tex
```

La fuente es `manuscript/preprint_es.tex`; las referencias están en
`manuscript/references.bib` y las figuras en `manuscript/figures/`.

## 6. Qué se versiona y qué se archiva

Se versiona en Git:

- código (`src/`), scripts, tests, `configs/`, `conftest.py`, `requirements.txt`;
- `README.md`, `REPRODUCING.md`, `manuscript/`;
- artefactos curados: `.md`, `.json`, `.png` y `scores_*.npz` de `reports/`;
- `pip-freeze.txt`, `environment.txt`, `CITATION.cff`, licencias.

No se versiona:

- `.venv/`, `__pycache__/`, `.pytest_cache/`;
- datasets `.npz` grandes, checkpoints `.pt` y el caché `P_ref_*.npy`;
- directorios de datos crudos o procesados.

El dataset derivado completo (~25 MB de `reports/`) se archiva aparte en
Zenodo/OSF si se requiere reproducibilidad byte a byte. En caso contrario, la
unidad de reproducibilidad es el pipeline con la versión de CircAdapt, el
`pip-freeze.txt` y las semillas.

## 7. Checklist de release

- [x] Checkout limpio reproduce las tablas 1–3 dentro del redondeo.
- [x] `pip-freeze.txt` y `environment.txt` versionados.
- [x] `CIRCADAPT_VERSION` fijado y registrado.
- [x] `git status` limpio; sin secretos, claves ni archivos personales.
- [x] Crear tag `v1.1.0`, release en GitHub y archivo en Zenodo.
- [x] Añadir el DOI resultante a `README.md`, `CITATION.cff` y el manuscrito.
      DOI de la versión: `10.5281/zenodo.24505901`
      (concept DOI: `10.5281/zenodo.24487582`).
