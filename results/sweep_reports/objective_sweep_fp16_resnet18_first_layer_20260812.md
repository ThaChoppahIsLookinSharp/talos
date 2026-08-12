# Objective sweep: FP16 y primera capa de ResNet18

Inicio: 2026-08-12 08:21:35 CEST

Fin: 2026-08-12 09:15:24 CEST

Duracion: 53 min 49 s

Commit: `2929d89`

## Estado

El manifest contiene 7/7 casos con codigo de retorno `0`. Todos los
casos produjeron soluciones y no quedaron procesos del sweep vivos.

Artefactos:

- Resultados:
  `results/objective_sweep_fp16_resnet18_first_layer/20260812_082135`
- Manifest:
  `results/objective_sweep_fp16_resnet18_first_layer/20260812_082135/manifest.csv`
- Log del run:
  `results/objective_sweep_fp16_resnet18_first_layer/run_20260812_082135.log`
- Log del monitor:
  `results/objective_sweep_fp16_resnet18_first_layer/monitor_20260812_082135.log`

El monitor quedo configurado con un periodo de cinco horas. El sweep
termino antes del primer periodo, por lo que registro el arranque, una
comprobacion intermedia y el cierre, pero no vencio ningun control de
cinco horas.

Configuracion:

- Workload: `workloads/resnet18_first_layer.onnx`
- Pool: `configs/ip_pool_fp16_65nm.yaml`
- Level 1: NSGA-II, poblacion 80, 8 generaciones, 8 workers
- Level 2: exhaustivo, hasta 100000 combinaciones
- Arquitecturas transferidas por caso: hasta 12
- Restricciones de usuario: ninguna
- Frecuencia de referencia: 200 MHz
- Seed: 1

## Pool FP16

El pool conserva unicamente los dos PEs FP16 de la caracterizacion
recibida. Las areas originales en um2 se convirtieron a mm2.

| PE | formato | area mm2 | fmax MHz | idle W | active W | pJ/MAC |
|---|---|---:|---:|---:|---:|---:|
| `pe_sauria_fp16` | float16 | 0.00547884 | 200.000 | 0.00112536 | 0.00177765 | 3.26145 |
| `pe_fpnew_fp16_fma` | float16 | 0.00467244 | 200.080 | 0.00106090 | 0.00160547 | 2.72285 |

Tambien incluye los 12 RFs y los 3 GBs caracterizados del pool de
origen. El unico bloque sintetico es `dram_proxy_512b`, necesario para
el flujo completo; su energia activa se recalibra con CACTI en cada
caso.

## Representacion del workload

El nuevo ONNX contiene una capa `Conv`. Su entrada, pesos, bias y salida
visible son `FLOAT`, por lo que Talos la entrega a ZigZag como
`float32/32` para `I`, `W` y `O`.

Esto no coincide con los PEs `float16/16`. Talos emitio un warning
agregado por caso y conservo ambos PEs, de acuerdo con la politica
actual. No hubo cuantizacion automatica. Por tanto, las cifras combinan
el mapping FP32 del workload con el PPA de PEs FP16 y deben leerse como
una exploracion arquitectural del pool, no como una implementacion
fisicamente compatible del ONNX sin conversion previa a FP16.

## Resultados

Los minimos de cada columna pueden pertenecer a filas diferentes.

| caso | implementaciones | arq. L1 | PEs | area min mm2 | power min W | energia min J | latencia min s | max inferencias/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| energy | 304 | 12 | 128-256 | 19.8098 | 4.14106 | 0.0311455 | 0.00462830 | 216.06 |
| area | 24208 | 12 | 16 | **0.451629** | 0.651002 | 0.222722 | 0.100352 | 9.96 |
| performance | 416 | 12 | 1024 | 81.7246 | 21.3171 | 0.0281267 | **0.00131938** | **757.93** |
| energy_area | 9696 | 12 | 16-128 | 0.546427 | **0.293842** | **0.0208520** | 0.00925432 | 108.06 |
| area_performance | 2576 | 12 | 32-1024 | 2.55389 | 2.58886 | 0.0278925 | 0.00131941 | 757.92 |
| energy_performance | 768 | 12 | 1024 | 80.4586 | 18.4478 | 0.0244309 | 0.00131947 | 757.88 |
| energy_area_performance | 7632 | 12 | 16-1024 | 0.546427 | 0.610657 | 0.0311455 | 0.00131940 | 757.92 |

En total hay 45600 filas validas y 42336 implementaciones fisicas
unicas, considerando la arquitectura y los IPs seleccionados.

Los extremos globales son:

- Menor area: `area`, 16 PEs, `0.45162864 mm2`.
- Menor potencia media: `energy_area`, 16 PEs, `0.293842449 W`.
- Menor energia: `energy_area`, 64 PEs, `0.0208519983 J`.
- Menor latencia: `performance`, 1024 PEs, `0.001319375 s`,
  equivalente a `757.93 inferencias/s`.

Todos estos extremos seleccionan `pe_fpnew_fp16_fma`. Frente a
`pe_sauria_fp16`, este PE tiene un 14.72% menos de area, un 9.69% menos
de potencia activa y un 16.51% menos de energia por MAC, con una
frecuencia practicamente igual. Su seleccion en los extremos es
coherente con la caracterizacion del pool.

El minimo del caso `energy` queda un 49.36% por encima del minimo
encontrado por `energy_area`. Cada caso ejecuta una busqueda Level 1
independiente y estocastica; el nombre del caso indica sus objetivos,
pero no certifica un optimo global compartido entre ejecuciones.

## Validacion

- Los 45600 resultados reportan `level2_valid=True` y
  `constraints_satisfied=True`.
- Todas las latencias cumplen exactamente:
  `cycles / (reference_frequency_mhz * 1e6)`.
- Todos los resultados cumplen exactamente:
  `energy = power * latency`.
- Todos operan a 200 MHz y 1.2 V. El fmax fisico esta entre 200 y
  200.080032 MHz.
- Las dos opciones de PE aparecen en el exhaustivo: 22800 filas por PE.
- Solo se seleccionaron los dos PEs FP16 del nuevo pool.

## Comando

```bash
.venv/bin/python -u examples/objective_sweep.py \
  --workload workloads/resnet18_first_layer.onnx \
  --ip-pool configs/ip_pool_fp16_65nm.yaml \
  --results-dir results/objective_sweep_fp16_resnet18_first_layer \
  --level1-pop-size 80 \
  --level1-generations 8 \
  --workers 8 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 12 \
  --no-constraints \
  --seed 1
```
