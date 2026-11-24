#!/usr/bin/env bash
# Reproduce la corrida primaria completa del estudio Phase 0.
#
# Uso:
#   bash scripts/reproduce_all.sh              # 8 hilos por defecto
#   THREADS=10 bash scripts/reproduce_all.sh
#
# Requiere haber creado .venv e instalado requirements.txt, torch, lightgbm y
# pyCircAdapt (ver REPRODUCING.md). Genera ~25 MB en reports/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
THREADS="${THREADS:-8}"

if [[ ! -x "$PY" ]]; then
  echo "No existe $PY. Creá el entorno virtual (ver REPRODUCING.md)." >&2
  exit 1
fi

if [[ -e "$ROOT/reports/benchmark_dataset_bench500_imp.npz" ]]; then
  echo "Ya existe reports/benchmark_dataset_bench500_imp.npz." >&2
  echo "Borralo o cambiá el tag si querés regenerar desde cero." >&2
  exit 1
fi

step() { date "+[%Y-%m-%d %H:%M:%S] $*"; }

step "1/6 dataset + benchmark lineal/CNN"
"$PY" scripts/run_benchmark.py \
  --patients 500 --signal-length 256 --tag bench500_imp --threads "$THREADS"

step "2/6 escalera de modelos"
"$PY" scripts/run_gbt_benchmark.py \
  --dataset-tag bench500_imp --out-tag ladder500_imp_reconcile \
  --patient-normalize none --threads "$THREADS"

step "3/6 control negativo"
"$PY" scripts/negative_control.py --tag bench500_imp --seed 0

step "4/6 curva de aprendizaje"
"$PY" scripts/benchmark_learning_curve.py --tag bench500_imp --threads "$THREADS"

step "5/6 protocolo de dominio desplazado sellado"
"$PY" scripts/reconcile_shift_protocol.py

step "6/6 figuras del manuscrito"
"$PY" scripts/make_manuscript_figures.py

step "Listo. Artefactos en reports/ y manuscript/figures/."
