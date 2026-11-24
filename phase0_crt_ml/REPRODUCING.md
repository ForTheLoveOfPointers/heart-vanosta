# Reproducción del estudio Phase 0 in silico

Este documento es la contraparte de reproducibilidad de
[`../manuscript/preprint_es.tex`](../manuscript/preprint_es.tex). Registra el
entorno exacto, los comandos, las semillas y los artefactos esperados de la
ejecución principal.

## 1. Entorno verificado

Ejecución usada para regenerar los resultados del manuscrito:

- **Sistema:** macOS 15.6 (Darwin, arm64), Apple M2, 8 núcleos, 24 GB de RAM.
- **Python:** 3.12.16 (Homebrew).
- **pyCircAdapt:** 2602 (distribución binaria
  `circadapt-2602-py3-none-macosx_12_0_universal2.whl`).
- **numpy:** 2.5.5 · **scipy:** 1.18.1 · **matplotlib:** 3.11.2 ·
  **PyYAML:** 6.0.4.
- **torch:** 2.15.0.
- **lightgbm:** 4.7.0, compilado en macOS con `USE_OPENMP=OFF` (véase la nota de
  macOS más abajo).
- **Semilla principal:** `0` (configuración `seed: 0`; banco de pruebas
  `--seed 0`).
- **Partición:** 350 pacientes de entrenamiento / 75 de validación / 75 de
  prueba.
- **Etiqueta del conjunto de datos:** `bench500_imp`. **Etiqueta de la escalera:**
  `ladder500_imp_reconcile`.

Para regenerar el registro del entorno:

```bash
python3.12 --version
sw_vers
uname -m
git rev-parse HEAD
.venv/bin/pip freeze > pip-freeze.txt
.venv/bin/python -c "import circadapt; print(getattr(circadapt, '__version__', '?'))"
```

El archivo `pip-freeze.txt` se versiona junto con la versión publicada. Los
archivos `.npz` grandes y los puntos de control no se versionan: se regeneran con
el flujo de trabajo.

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

`setup_circadapt.sh` instala la distribución binaria de CircAdapt desde el sitio
del proyecto. Se puede fijar la versión y verificar el checksum:

```bash
CIRCADAPT_VERSION=2602 CIRCADAPT_SHA256=<sha256-del-wheel> \
  bash scripts/setup_circadapt.sh
```

Resultado esperado de la prueba mínima:

```text
SMOKE OK
samples=27 features=42
```

Y de la suite de pruebas:

```text
23 passed
```

### Nota para macOS: conflicto de `libomp` entre torch y LightGBM

Las distribuciones binarias de `torch` y `lightgbm` para macOS incluyen cada una
su propia copia de OpenMP. Cargarlas en el mismo proceso aborta con:

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Solución usada en la ejecución reportada:

```bash
brew install libomp
.venv/bin/pip install --no-binary lightgbm \
  --config-settings=cmake.define.USE_OPENMP=OFF "lightgbm==4.7.0"
```

En Linux, `pip install lightgbm` funciona sin este paso.

## 3. Análisis principal

La línea base principal es la ruta **sin normalización intrapaciente**
(`patient_normalize: []`, `features.mode: raw`). El banco de pruebas lineal/CNN
crea el conjunto de datos `bench500_imp` que reutilizan los pasos siguientes.
**No** utilice `--skip-cnn`: la escalera de modelos necesita los embeddings de la
CNN.

Para ejecutar todo de una vez:

```bash
THREADS=8 bash scripts/reproduce_all.sh
```

De forma equivalente, paso a paso:

```bash
# 1. Conjunto de datos de 500 pacientes + banco de pruebas lineal y CNN
#    (incluye el dominio desplazado sellado)
.venv/bin/python scripts/run_benchmark.py \
  --patients 500 --signal-length 256 --tag bench500_imp --threads 8

# 2. Escalera: LambdaMART, embeddings de la CNN, híbridos, mezcla y control negativo
.venv/bin/python scripts/run_gbt_benchmark.py \
  --dataset-tag bench500_imp \
  --out-tag ladder500_imp_reconcile \
  --patient-normalize none --threads 8

# 3. Curva de aprendizaje
.venv/bin/python scripts/benchmark_learning_curve.py \
  --tag bench500_imp --threads 8

# 4. Reconciliación del protocolo sellado al 2 % y al 5 %
.venv/bin/python scripts/reconcile_shift_protocol.py

# 5. Figuras del manuscrito
.venv/bin/python scripts/make_manuscript_figures.py
```

Tiempo aproximado: 1 hora en un Apple M2 de 8 núcleos. Espacio en disco:
~25 MB en `reports/`.

## 4. Salidas esperadas

Artefactos principales del manuscrito:

| Artefacto | Origen |
|---|---|
| Conjunto de datos, particiones y arreglos de origen/desplazado | `reports/benchmark_dataset_bench500_imp.npz`, `reports/benchmark_shifted_bench500_imp.npz`, `reports/benchmark_split_bench500_imp.json` |
| Resumen lineal/CNN | `reports/benchmark_summary_bench500_imp.md` |
| Escalera + intervalos de confianza + mezcla (principal) | `reports/gbt_report_ladder500_imp_reconcile.json`, `reports/gbt_summary_ladder500_imp_reconcile.{md,png}` |
| Puntuaciones por modelo (auditoría) | `reports/scores_ladder500_imp_reconcile.npz` |
| Protocolo sellado al 2 % y al 5 % | `reports/reconcile_shift_protocol.json` |
| Curva de aprendizaje | `reports/benchmark_learning_curve_bench500_imp.{json,png}` |
| Figuras del manuscrito | `manuscript/figures/{response_surface,model_comparison,learning_curve,ablations}.png` |

**Advertencia de procedencia.** `reports/pitch_summary_bench500_pp.md` es un
artefacto histórico de una ejecución exploratoria y no forma parte de esta
versión. Los valores principales son los de la ejecución
`ladder500_imp_reconcile`; `reports/benchmark_summary_bench500_imp.md`
corresponde a la misma ejecución principal sin normalización.

El control negativo citado en el manuscrito es el de la escalera (GBT con
etiquetas barajadas: 0.080 y Spearman −0.149); se genera dentro de
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

- código (`src/`), scripts, pruebas, `configs/`, `conftest.py`,
  `requirements.txt`;
- `README.md`, `REPRODUCING.md`, `manuscript/`;
- artefactos curados: `.md`, `.json`, `.png` y `scores_*.npz` de `reports/`;
- `pip-freeze.txt`, `environment.txt`, `CITATION.cff` y licencias.

No se versiona:

- `.venv/`, `__pycache__/`, `.pytest_cache/`;
- conjuntos de datos `.npz` grandes, puntos de control `.pt` y la caché
  `P_ref_*.npy`;
- directorios de datos crudos o procesados.

El conjunto de datos derivado completo (~25 MB en `reports/`) se archiva aparte
en Zenodo/OSF si se requiere reproducibilidad byte a byte. En caso contrario, la
unidad de reproducibilidad es el flujo de trabajo con la versión de CircAdapt, el
`pip-freeze.txt` y las semillas.

## 7. Lista de verificación de la versión

- [x] Una copia limpia del repositorio reproduce las tablas 1–3 dentro del
      redondeo.
- [x] `pip-freeze.txt` y `environment.txt` versionados.
- [x] `CIRCADAPT_VERSION` fijado y registrado.
- [x] `git status` limpio; sin secretos, claves ni archivos personales.
- [x] Crear la etiqueta de la versión, publicar la versión en GitHub y archivar
      en Zenodo.
- [x] Añadir el DOI resultante a `README.md`, `CITATION.cff` y el manuscrito.
      DOI de la versión: véase `CITATION.cff`.
      (DOI conceptual: `10.5281/zenodo.24487582`).
