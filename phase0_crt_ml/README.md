# Phase 0 — Aprendibilidad in silico de la optimización AV/VV en TRC usando impedancia intracardíaca

Este repositorio implementa un estudio de simulación para estimar si una señal
de impedancia intracardíaca, junto con contexto del dispositivo (frecuencia
cardíaca, AV intrínseco y QRS), permite elegir la combinación AV/VV que maximiza
el trabajo sistólico en terapia de resincronización cardíaca (TRC).

El proyecto es deliberadamente **impedance-only**: no simula EGM ni
acelerómetro.

> Documentación general del repositorio, licencias y estado del manuscrito:
> [`../README.md`](../README.md). Entorno exacto, comandos y salidas esperadas:
> [`REPRODUCING.md`](REPRODUCING.md).

## Qué hace el pipeline

1. Genera pacientes virtuales con CircAdapt (`VanOsta2024`).
2. Barre una grilla AV × VV por paciente.
3. Simula una señal de impedancia intracardíaca con ruido, deriva, ganancia y
   offset variables.
4. Extrae 42 features interpretables de impedancia + contexto + programación.
5. Entrena un ranker lineal pairwise para ordenar programaciones.
6. Evalúa en dominio fuente y en un dominio de impedancia desplazado.
7. Deja la normalización within-patient como análisis de sensibilidad opcional.

La regla central se mantiene: la fisiología y las etiquetas nunca ven los
parámetros de ruido del sensor.

## Estructura

```text
src/phase0/
├── benchmark.py          # dataset con impedancia cruda y features
├── dataset.py            # dataset simple para el experimento inicial
├── evaluate.py           # métricas, splits, ablaciones
├── features.py           # 42 features de impedancia + contexto
├── models/               # ranker pairwise, LambdaMART y CNN
├── physiology/           # adaptador CircAdapt
├── sensors/
│   ├── domain.py         # ruido/ganancia/offset/deriva de impedancia
│   ├── impedance.py      # modelo de observación de impedancia
│   └── pipeline.py       # observación y resampleo 1-canal
└── types.py
scripts/
├── run_benchmark.py            # dataset + baseline linear/CNN
├── run_gbt_benchmark.py        # escalera LambdaMART/híbridos/blend
├── negative_control.py         # control con etiquetas barajadas
├── benchmark_learning_curve.py # curva de aprendizaje
├── reconcile_shift_protocol.py # protocolo de dominio desplazado sellado
├── make_manuscript_figures.py  # figuras del manuscrito
├── analyze_benchmark.py
├── analyze_linear_benchmark.py
├── run_phase0.py
├── smoke_test.py
└── setup_circadapt.sh
```

## Instalación

Requiere **Python 3.12**. CircAdapt oficialmente soporta hasta 3.12.

```bash
cd phase0_crt_ml
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest torch lightgbm
bash scripts/setup_circadapt.sh
.venv/bin/python scripts/smoke_test.py
```

CircAdapt se instala con:

```bash
bash scripts/setup_circadapt.sh
```

En macOS, `torch` y `lightgbm` cargan dos copias de `libomp` y pueden abortar
con `OMP: Error #15`. Compilá LightGBM sin OpenMP:

```bash
brew install libomp
.venv/bin/pip install --no-binary lightgbm \
  --config-settings=cmake.define.USE_OPENMP=OFF "lightgbm==4.7.0"
```

## Uso rápido

```bash
# verificación mínima
.venv/bin/python scripts/smoke_test.py

# baseline primario: 500 pacientes, impedancia only, sin normalización.
# No usar --skip-cnn: el ladder necesita los embeddings de la CNN.
.venv/bin/python scripts/run_benchmark.py \
  --patients 500 --signal-length 256 --tag bench500_imp --threads 10
```

La corrida completa (dataset, escalera de modelos, control negativo, curva de
aprendizaje y reconciliación) está documentada paso a paso en
[`REPRODUCING.md`](REPRODUCING.md) y se puede lanzar con:

```bash
bash scripts/reproduce_all.sh
```

## Configuración

`configs/default.yaml` define:

- 500 pacientes de benchmark, 8 AV × 7 VV = 56 programaciones por paciente;
- 3 beats por programación;
- grilla AV 80–300 ms y VV −60 a +120 ms;
- `features.mode: raw` y `patient_normalize: []` (baseline primario);
- normalización within-patient `[imp_]` solo como sensibilidad;
- dominio fuente y dominio desplazado solo con parámetros de impedancia.

## Resultados: baseline primario sin normalización (500 pacientes)

Estos son los números del **baseline primario** (`bench500_imp`, sin
normalización within-patient, gates 3 latidos, grilla 8 AV × 7 VV). Split:
350 train / 75 validation / 75 test; total 28.000 latidos. La normalización
within-patient es transductiva y queda como análisis de sensibilidad.

| Modelo | Casi óptimo source | Regret source | Casi óptimo shifted | Regret shifted |
|---|---:|---:|---:|---:|
| Linear (impedancia + contexto) | 0.640 | **0.0180** | 0.640 | **0.0217** |
| Linear sin sensores de onda | 0.493 | 0.0366 | 0.493 | 0.0366 |
| LambdaMART sobre features | 0.680 | 0.0248 | 0.600 | 0.0317 |
| Tiny CNN sobre señal cruda | 0.107 | 0.1561 | 0.107 | 0.1467 |
| LambdaMART sobre embeddings CNN | 0.667 | 0.0240 | 0.507 | 0.0361 |
| LambdaMART híbrido (features + embeddings) | 0.693 | 0.0236 | 0.587 | 0.0248 |
| Linear sobre features + embeddings CNN | **0.800** | **0.0109** | **0.680** | **0.0205** |
| Blend ponderado por validación | **0.800** | **0.0109** | **0.680** | **0.0205** |

Lectura:

- El GBT (LambdaMART) **no supera** al ranker linear con features diseñadas:
  su sesgo inductivo (superficie cuadrática AV/VV + interacciones de contexto)
  coincide con el simulador. El GBT gana en argmax exacto, pero pierde en
  Spearman, regret y empeora más bajo *sensor shift*.
- La CNN sola es mala (como se esperaba), pero sus **embeddings** aportan señal
  complementaria: el linear sobre features + embeddings es el mejor modelo
  in-domain (**0.800** casi óptimo, regret **0.0109**).
- El **blend** (peso elegido en validación, nunca en test) seleccionó el modelo
  con embeddings, así que coincide con él (**0.800 / 0.0109** in-domain;
  **0.680 / 0.0205** bajo shift).
- Bootstrap pareado blend − linear (test): casi óptimo **+0.160** [+0.053,
  +0.267]; regret **−0.0072** [−0.0119, −0.0029] (in-domain). Bajo shift la
  mejora no es significativa (+0.040 [−0.067, +0.147]).
- Control negativo (GBT con etiquetas barajadas intra-paciente): casi óptimo
  0.080, Spearman −0.149 → no hay fuga de información.

La tabla usa el umbral estricto del **2 %** (estándar en la literatura de
ranking). El endpoint del piloto es más permisivo —dentro del **5 %** y 20 ms—,
así que también se evaluó a esa tolerancia: el modelo linear pasa de 0.880 a
0.867 entre el dominio de origen y el desplazado; el linear + embeddings, de
0.947 a 0.893; el blend coincide con este último; el piso sin sensores es 0.680
en ambos. Los valores exactos a 5 % se recalcularon en la corrida de reconciliación
(`reports/gbt_report_ladder500_imp_reconcile.json`) y en
`reports/reconcile_shift_protocol.json`. Es decir, bajo el desplazamiento extremo el modelo elige una
programación dentro del 5 % del óptimo en ~87–89 % de los pacientes, y la
pérdida se concentra en una cola (percentil 90 de pérdida 4–6 %).

Reproducir con:

```bash
.venv/bin/python scripts/run_gbt_benchmark.py --dataset-tag bench500_imp \
  --out-tag ladder500_imp_reconcile --patient-normalize none --threads 10
```

Salidas: `reports/gbt_report_ladder500_imp_reconcile.json`,
`reports/gbt_summary_ladder500_imp_reconcile.{md,png}`.

### Protocolo de dominio desplazado (test sellado)

El dominio desplazado es un **conjunto de validación externa sellado**: no se
usa para entrenar, ni para early stopping, ni para elegir rondas, pesos del
blend o features. Toda selección se hace solo con train/validation de origen,
siguiendo el protocolo estándar de WILDS/DomainBed y de validación externa
clínica (TRIPOD+AI). La cifra que se reporta como primaria es la del modelo
congelado en origen evaluado una sola vez en el dominio desplazado.

Como cota superior se reporta además un *bound* de adaptación (CV agrupada por
paciente dentro del dominio desplazado): 0.667 casi óptimo / 0.0214 regret,
prácticamente igual al modelo entrenado en origen (0.640 / 0.0217). La
normalización within-patient tampoco recupera la caída (0.627). Es decir, el
desplazamiento es principalmente **pérdida de información** (ruido y deriva),
no una transformación de calibración recuperable; entrenar con el dominio
desplazado sería fuga de información sin beneficio real. Si en el futuro se
construye una variante robusta, debe entrenarse con *domain randomization*
sobre un superconjunto y evaluarse en un tercer dominio desplazado no usado en
el entrenamiento (detalle en `RESEARCH_PLAN.md`, sección 5c).

## Qué prueba y qué no prueba

Prueba que, en este modelo sintético, la impedancia contiene información
suficiente para ordenar programaciones AV/VV mejor que el azar y mejor que el
contexto solo. La normalización within-patient es transductiva (requiere el
barrido completo de 56 programaciones) y se reporta solo como análisis de
sensibilidad: no recupera la caída bajo el desplazamiento y no forma parte del
baseline de un solo latido.

No prueba que la impedancia real de un paciente permita elegir AV/VV con
precisión clínica. El modelo de impedancia es fenomenológico y el óptimo VV está
influido por una penalización de resincronización. La validación requiere un
estudio prospectivo con impedancia intracardíaca real y una referencia
hemodinámica.

## Limitaciones

- Todos los pacientes y señales son simulados.
- El dominio desplazado es un stress test sintético, no una cohorte real.
- No se modelan EGM ni acelerómetro; el proyecto es impedance-only por diseño.
- La impedancia real depende de posición del electrodo, hematocrito, postura,
  respiración y fibrosis; el modelo no captura toda esa variabilidad.
- El óptimo VV sigue siendo la parte más débil del simulador.
- La normalización within-patient no es aplicable a decisiones con un solo
  latido: requiere conocer todas las programaciones candidatas del paciente.

## Próximos pasos

1. ~~Correr el baseline primario sin normalización~~ (hecho: ver tabla anterior).
2. Validar el modelo de impedancia con datos agudos reales.
3. Probar el blend linear+embeddings con un encoder entrenado en más datos;
   congelar un modelo interpretable (16–26 features + embedding reducido).
4. Pilotear en 30–40 pacientes con barrido AV/VV y referencia invasiva o eco.
5. Recién después, diseñar un estudio aleatorizado de optimización.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Referencias clave

- Bocchiardo et al., *Europace* 2010: optimización de AV/VV con impedancia
  intracardíaca.
- Ginks et al., *Europace* 2011: modelo patient-specific de impedancia para
  predecir el óptimo AV/VV.
- Sciaraffia et al., *J Interv Card Electrophysiol* 2013: necesidad de
  calibración patient-specific.
- Delnoy et al., *Indian Pacing Electrophysiol J* 2021 (BIO.Detect HF II):
  stroke impedance y VTI aórtico.
- Knijnik et al., *Heart Rhythm O2* 2023: meta-análisis de algoritmos
  automáticos de AV/VV sin beneficio clínico duro.
