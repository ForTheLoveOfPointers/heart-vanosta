# Phase 0 — Aprendibilidad in silico de la optimización AV/VV en TRC usando impedancia intracardíaca

Este repositorio implementa un estudio de simulación para estimar si una señal
de impedancia intracardíaca, junto con el contexto del dispositivo (frecuencia
cardíaca, AV intrínseco y QRS), permite elegir la combinación AV/VV que maximiza
el trabajo sistólico en terapia de resincronización cardíaca (TRC).

El proyecto se limita deliberadamente a la impedancia: no simula EGM ni
acelerómetro.

> Documentación general del repositorio, licencias y estado del manuscrito:
> [`../README.md`](../README.md). Entorno exacto, comandos y salidas esperadas:
> [`REPRODUCING.md`](REPRODUCING.md).

## Qué hace el flujo de trabajo

1. Genera pacientes virtuales con CircAdapt (`VanOsta2024`).
2. Recorre una rejilla AV × VV por paciente.
3. Simula una señal de impedancia intracardíaca con ruido, deriva, ganancia y
   desplazamiento regenerados en cada latido (modelo de observación
   deliberadamente adverso).
4. Extrae 42 características interpretables de impedancia, contexto y
   programación.
5. Entrena un modelo de ordenación lineal por pares para ordenar programaciones.
6. Evalúa en el dominio de origen y en un dominio de impedancia desplazado.
7. Deja la normalización intrapaciente como análisis de sensibilidad opcional.

La regla central se mantiene: la fisiología y las etiquetas nunca ven los
parámetros de ruido del sensor.

## Estructura

```text
src/phase0/
├── benchmark.py          # conjunto de datos con impedancia cruda y características
├── evaluate.py           # métricas y particiones
├── features.py           # 42 características de impedancia + contexto
├── models/               # ordenación por pares, LambdaMART y CNN
├── physiology/           # adaptador CircAdapt
├── sensors/
│   ├── domain.py         # ruido/ganancia/desplazamiento/deriva de impedancia
│   ├── impedance.py      # modelo de observación de impedancia
│   └── pipeline.py       # observación y remuestreo monocanal
└── types.py
scripts/
├── run_benchmark.py            # conjunto de datos + línea base lineal/CNN
├── run_gbt_benchmark.py        # escalera LambdaMART/híbridos/mezcla
├── benchmark_learning_curve.py # curva de aprendizaje
├── reconcile_shift_protocol.py # protocolo de dominio desplazado sellado
├── make_manuscript_figures.py  # figuras del manuscrito
├── smoke_test.py               # verificación mínima de CircAdapt
└── setup_circadapt.sh
```

## Instalación

Requiere **Python 3.12**. CircAdapt admite oficialmente hasta la versión 3.12.

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
con `OMP: Error #15`. Compile LightGBM sin OpenMP:

```bash
brew install libomp
.venv/bin/pip install --no-binary lightgbm \
  --config-settings=cmake.define.USE_OPENMP=OFF "lightgbm==4.7.0"
```

## Uso rápido

```bash
# verificación mínima
.venv/bin/python scripts/smoke_test.py

# línea base principal: 500 pacientes, solo impedancia, sin normalización.
# No usar --skip-cnn: la escalera necesita los embeddings de la CNN.
.venv/bin/python scripts/run_benchmark.py \
  --patients 500 --signal-length 256 --tag bench500_imp --threads 10
```

La ejecución completa (conjunto de datos, escalera de modelos, control negativo,
curva de aprendizaje y reconciliación) está documentada paso a paso en
[`REPRODUCING.md`](REPRODUCING.md) y se puede lanzar con:

```bash
bash scripts/reproduce_all.sh
```

## Configuración

`configs/default.yaml` define:

- 500 pacientes del banco de pruebas, 8 AV × 7 VV = 56 programaciones por
  paciente;
- 3 latidos por programación;
- rejilla AV de 80–300 ms y VV de −60 a +120 ms;
- `features.mode: raw` y `patient_normalize: []` (línea base principal);
- normalización intrapaciente `[imp_]` solo como análisis de sensibilidad;
- dominio de origen y dominio desplazado solo con parámetros de impedancia.

## Resultados: línea base principal sin normalización (500 pacientes)

Estos son los valores de la **línea base principal** (`bench500_imp`, sin
normalización intrapaciente, 3 latidos por ajuste, rejilla de 8 AV × 7 VV). La
partición es de 350 pacientes de entrenamiento, 75 de validación y 75 de prueba;
en total, 28 000 latidos. La normalización intrapaciente es transductiva y queda
como análisis de sensibilidad.

| Modelo | Casi óptimo (origen) | Pérdida (origen) | Casi óptimo (desplazado) | Pérdida (desplazado) |
|---|---:|---:|---:|---:|
| Lineal (impedancia + contexto) | 0.640 | **0.0180** | 0.640 | **0.0217** |
| Lineal sin la onda de impedancia | 0.493 | 0.0366 | 0.493 | 0.0366 |
| LambdaMART sobre características | 0.680 | 0.0248 | 0.600 | 0.0317 |
| CNN pequeña sobre la señal cruda | 0.107 | 0.1561 | 0.107 | 0.1467 |
| LambdaMART sobre embeddings de la CNN | 0.667 | 0.0240 | 0.507 | 0.0361 |
| LambdaMART híbrido (características + embeddings) | 0.693 | 0.0236 | 0.587 | 0.0248 |
| Lineal sobre características + embeddings de la CNN | **0.800** | **0.0109** | **0.680** | **0.0205** |
| Mezcla ponderada por validación | **0.800** | **0.0109** | **0.680** | **0.0205** |

Lectura de los resultados:

- El GBT (LambdaMART) **no supera** al modelo de ordenación lineal con
  características diseñadas: su sesgo inductivo (superficie cuadrática AV/VV e
  interacciones de contexto) coincide con el simulador. El GBT obtiene mejores
  resultados en argmax exacto, pero peores en Spearman y pérdida, y empeora más
  bajo el desplazamiento del sensor.
- La CNN por sí sola es deficiente (como se esperaba), pero sus **embeddings**
  aportan señal complementaria: el modelo lineal sobre características más
  embeddings es el mejor en el dominio de origen (**0.800** casi óptimo, pérdida
  **0.0109**).
- La **mezcla** (peso elegido en validación, nunca en prueba) seleccionó el
  modelo con embeddings, de modo que coincide con él (**0.800 / 0.0109** en el
  dominio de origen; **0.680 / 0.0205** bajo el desplazamiento).
- Bootstrap pareado mezcla − lineal (prueba): casi óptimo **+0.160** [+0.053,
  +0.267]; pérdida **−0.0072** [−0.0119, −0.0029] (dominio de origen). Bajo el
  desplazamiento, la mejora no es significativa (+0.040 [−0.067, +0.147]).
- Control negativo (GBT con etiquetas barajadas dentro de cada paciente): casi
  óptimo 0.080, Spearman −0.149 → no hay fuga de información.

La tabla usa el umbral estricto del **2 %** (estándar en la literatura de
ordenación). El criterio de valoración del piloto es más permisivo —dentro del
**5 %** y 20 ms—, por lo que también se evaluó con esa tolerancia: el modelo
lineal pasa de 0.880 a 0.867 entre el dominio de origen y el desplazado; el
modelo lineal con embeddings, de 0.947 a 0.893; la mezcla coincide con este
último; el suelo sin la onda de impedancia es 0.680 en ambos dominios. Los
valores exactos al 5 % se recalcularon en la ejecución de reconciliación
(`reports/gbt_report_ladder500_imp_reconcile.json`) y en
`reports/reconcile_shift_protocol.json`. Es decir, bajo el desplazamiento extremo
el modelo elige una programación dentro del 5 % del óptimo en ~87–89 % de los
pacientes, y la pérdida se concentra en una cola (percentil 90 de pérdida 4–6 %).

Para reproducir estos resultados:

```bash
.venv/bin/python scripts/run_gbt_benchmark.py --dataset-tag bench500_imp \
  --out-tag ladder500_imp_reconcile --patient-normalize none --threads 10
```

Salidas: `reports/gbt_report_ladder500_imp_reconcile.json`,
`reports/gbt_summary_ladder500_imp_reconcile.{md,png}`.

### Protocolo de dominio desplazado (prueba sellada)

El dominio desplazado es un **conjunto de prueba externa sellado**: no se usa
para entrenar, ni para *early stopping*, ni para elegir rondas, pesos de la
mezcla o características. Toda la selección se realiza solo con datos de
entrenamiento y validación del dominio de origen, siguiendo el protocolo estándar
de WILDS/DomainBed y de validación externa clínica (TRIPOD+AI). El valor que se
reporta como principal es el del modelo congelado en origen, evaluado una sola vez
en el dominio desplazado.

Como cota superior se reporta además una cota de adaptación (validación cruzada
agrupada por paciente dentro del dominio desplazado): 0.667 casi óptimo / 0.0214
de pérdida, prácticamente igual al modelo entrenado en origen (0.640 / 0.0217).
La normalización intrapaciente tampoco recupera la caída: sobre el modelo
entrenado en origen da 0.560 al 2 %; adaptada al dominio de destino llega a 0.640,
sin superar de forma consistente la cota sin normalización. Es decir, el
desplazamiento es principalmente **pérdida de información** (más ruido y deriva,
con una calibración regenerada en cada latido) y no un desplazamiento fijo de
sesión que la estandarización pudiera deshacer; entrenar con el dominio desplazado
sería fuga de información sin beneficio real. Si en el futuro se construye una
variante robusta, debe entrenarse con *aleatorización de dominios* sobre un
superconjunto y evaluarse en un tercer dominio desplazado no usado en el
entrenamiento.

## Qué prueba y qué no prueba

Prueba que, en este modelo sintético, la impedancia contiene información
suficiente para ordenar programaciones AV/VV mejor que el azar y mejor que el
modelo sin onda (contexto y programación). La normalización intrapaciente es
transductiva (requiere el recorrido completo de 56 programaciones) y se reporta
solo como análisis de sensibilidad: no recupera la caída bajo el desplazamiento y
no forma parte de la línea base de un solo latido.

No prueba que la impedancia real de un paciente permita elegir AV/VV con
precisión clínica. El modelo de impedancia es fenomenológico y el óptimo VV está
influido por una penalización de resincronización. La validación requiere un
estudio prospectivo con impedancia intracardíaca real y una referencia
hemodinámica.

## Limitaciones

- Todos los pacientes y todas las señales son simulados.
- El dominio desplazado es una prueba de estrés sintética, no una cohorte real.
- No se modelan EGM ni acelerómetro; por diseño, el proyecto se limita a la
  impedancia.
- La impedancia real depende de la posición del electrodo, el hematocrito, la
  postura, la respiración y la fibrosis; el modelo no captura toda esa
  variabilidad.
- El óptimo VV sigue siendo el aspecto más débil del simulador.
- Las características de contexto (AV intrínseco, QRS y frecuencia cardíaca) son
  los valores verdaderos del simulador, no estimaciones del dispositivo; un
  dispositivo real las estimaría con error, de modo que el suelo sin la onda de
  impedancia es optimista.
- La normalización intrapaciente no es aplicable a decisiones con un solo latido:
  requiere conocer todas las programaciones candidatas del paciente.

## Próximos pasos

1. ~~Ejecutar la línea base principal sin normalización~~ (hecho: véase la tabla
   anterior).
2. Validar el modelo de impedancia con datos agudos reales.
3. Probar la mezcla lineal + embeddings con un codificador entrenado con más
   datos; congelar un modelo interpretable (16–26 características más una
   representación reducida).
4. Realizar un piloto con 30–40 pacientes, con recorrido AV/VV y referencia
   invasiva o ecocardiográfica.
5. Solo después, diseñar un estudio aleatorizado de optimización.

## Pruebas

```bash
.venv/bin/python -m pytest -q
```

## Referencias clave

- Bocchiardo et al., *Europace* 2010: optimización de AV/VV con impedancia
  intracardíaca.
- Ginks et al., *Europace* 2011: modelo específico del paciente para predecir el
  óptimo AV/VV a partir de la impedancia.
- Sciaraffia et al., *J Interv Card Electrophysiol* 2013: necesidad de
  calibración específica del paciente.
- Delnoy et al., *Indian Pacing Electrophysiol J* 2021 (BIO.Detect HF II):
  impedancia sistólica y VTI aórtico.
- Knijnik et al., *Heart Rhythm O2* 2023: metaanálisis de algoritmos automáticos
  de AV/VV sin beneficio clínico duro.
