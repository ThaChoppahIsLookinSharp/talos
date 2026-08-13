# Objective sweep: FP32 y primera capa de ResNet18

Inicio: 2026-08-13 22:27:34 CEST

Fin: 2026-08-13 23:18:01 CEST

Duracion: 50 min 27 s

Commit: `86909e2`

## Estado

El manifest contiene 7/7 casos con codigo de retorno `0`. Todos los
casos produjeron soluciones y no quedaron procesos del sweep vivos.

Artefactos:

- Base: `results/objective_sweep_fp32_resnet18_first_layer`
- Run: `20260813_222734`
- Manifest: `20260813_222734/manifest.csv`, bajo la base
- Log del run: `run_20260813_222734.log`, bajo la base
- Log del monitor: `monitor_20260813_222734.log`, bajo la base

El monitor quedo configurado con un periodo de cinco horas y
comprobacion del cierre cada minuto. El sweep termino antes del primer
periodo, por lo que registro el arranque y el cierre, pero no vencio
ningun control de cinco horas.

Configuracion:

- Workload: `workloads/resnet18_first_layer.onnx`
- Pool: `configs/ip_pool_fp32_65nm.yaml`
- Level 1: NSGA-II, poblacion 80, 8 generaciones, 16 workers
- Level 2: exhaustivo, hasta 100000 combinaciones
- Arquitecturas transferidas por caso: hasta 12
- Restricciones de usuario: ninguna
- Frecuencia de referencia: 200 MHz
- Seed: 1

## Pool FP32

El pool conserva los tres PEs FP32 validos de la caracterizacion
recibida. Las areas originales en um2 se convirtieron a mm2.

| PE | area mm2 | fmax MHz | active W | pJ/MAC |
|---|---:|---:|---:|---:|
| `pe_sauria_fp32` | 0.0141962 | 200.000 | 0.004750 | 8.85630 |
| `pe_fpnew_fp32_fma` | 0.0122879 | 200.000 | 0.004116 | 8.46935 |
| `pe_dawsonjon_fp32_mac` | 0.0134291 | 200.924 | 0.003326 | 15.9123 |

Tambien incluye todas las memorias con caracterizacion numerica
valida:
12 RFs y 3 GBs. Las tres variantes OpenEye de doble puerto no tienen
informes PPA completos y la SRAM PULP es un black box sin macro
TSMC65. Estas cuatro entradas no se pueden convertir en alternativas
numericas sin inventar valores.

El unico bloque sintetico es `dram_proxy_512b`, necesario para el
flujo completo. Su energia activa se recalibra con CACTI en cada caso.

## Representacion del workload

El ONNX contiene una capa `Conv`. Su entrada, pesos, bias y salida
visible son `FLOAT`, por lo que Talos la entrega a ZigZag como
`float32/32` para `I`, `W` y `O`.

Los tres PEs declaran `float32/32`. No hubo warning de precision ni
cuantizacion automatica, por lo que esta run si mantiene coherencia
entre la representacion visible del workload y los PEs.

## Resultados

Los minimos de cada columna pueden pertenecer a filas diferentes.

| caso | impl. | arq. L1 | PEs |
|---|---:|---:|---:|
| energy | 304 | 12 | 128-256 |
| area | 24208 | 12 | 16 |
| performance | 416 | 12 | 1024 |
| energy_area | 9696 | 12 | 16-128 |
| area_performance | 2576 | 12 | 32-1024 |
| energy_performance | 768 | 12 | 1024 |
| energy_area_performance | 7632 | 12 | 16-1024 |

| caso | area min mm2 | power min W | energia min J |
|---|---:|---:|---:|
| energy | 20.7846 | 4.46235 | 0.0341195 |
| area | **0.573476** | 0.686565 | 0.230716 |
| performance | 89.5228 | 23.7403 | 0.0313240 |
| energy_area | 0.668274 | **0.329406** | **0.0243084** |
| area_performance | 2.79759 | 2.74951 | 0.0321480 |
| energy_performance | 88.2568 | 20.8710 | 0.0276400 |
| energy_area_performance | 0.668274 | 0.686382 | 0.0341195 |

| caso | latencia min s | max inferencias/s |
|---|---:|---:|
| energy | 0.00462830 | 216.06 |
| area | 0.100352 | 9.96 |
| performance | **0.00131938** | **757.93** |
| energy_area | 0.00925432 | 108.06 |
| area_performance | 0.00131941 | 757.92 |
| energy_performance | 0.00131947 | 757.88 |
| energy_area_performance | 0.00131940 | 757.92 |

En total hay 45600 filas validas y 42336 implementaciones fisicas
unicas, considerando la arquitectura y los IPs seleccionados.

Los extremos globales son:

- Menor area: `area`, 16 PEs, `0.573476 mm2`.
- Menor potencia media: `energy_area`, 16 PEs, `0.329406049 W`.
- Menor energia: `energy_area`, 64 PEs, `0.0243083805 J`.
- Menor latencia: `performance`, 1024 PEs, `0.001319375 s`,
  equivalente a `757.93 inferencias/s`.

Todos estos extremos seleccionan `pe_fpnew_fp32_fma`. Frente a
`pe_sauria_fp32`, este PE tiene un 13.44% menos de area, un 13.36%
menos de potencia activa y un 4.37% menos de energia por MAC, con la
misma frecuencia. Su seleccion es coherente con la caracterizacion.

## PE Dawsonjon

El PE Dawsonjon es una implementacion serializada. Su energia
reportada cumple exactamente:

```text
15.9123 pJ/MAC = (3.32649 - 3.22383) mW * 31 / 200 MHz
```

Por tanto, se modelo con `1/31 MAC/ciclo`, no con un MAC por ciclo. El
exhaustivo considero 68400 combinaciones y conservo 45600. Las 22800
restantes son exactamente el tercio que seleccionaba Dawsonjon:
ninguna alcanzo la capacidad requerida por los mappings congelados.
Mantenerlo en el pool conserva la caracterizacion sin falsear su
throughput.

## Comparacion con el sweep FP16

Ambos sweeps usan el mismo workload, seed, presupuesto Level 1,
arquitecturas transferidas y Level 2 exhaustivo. FP32 cambia los
mejores valores globales respecto a FP16 de esta forma:

| metrica | FP16 | FP32 | cambio FP32 |
|---|---:|---:|---:|
| area mm2 | 0.451629 | 0.573476 | +26.98% |
| power W | 0.293842 | 0.329406 | +12.10% |
| energia J | 0.0208520 | 0.0243084 | +16.58% |
| latencia s | 0.00131938 | 0.00131938 | 0% |
| inferencias/s | 757.93 | 757.93 | 0% |

La latencia no cambia porque ambos pools operan a 200 MHz y reutilizan
el mismo mapping. Area, potencia y energia si reflejan la
caracterizacion fisica del PE seleccionado.

La comparacion FP16 es descriptiva: aquel sweep conservo PEs FP16 para
un ONNX FP32 por politica. La run actual es la que mantiene una
representacion coherente de extremo a extremo.

## Validacion

- Los 45600 resultados reportan `level2_valid=True` y
  `constraints_satisfied=True`.
- Todas las latencias cumplen exactamente:
  `cycles / (reference_frequency_mhz * 1e6)`.
- Todos los resultados cumplen exactamente:
  `energy = power * latency`.
- Todos operan a 200 MHz y 1.2 V, con fmax fisico de 200 MHz.
- FPnew y SAURIA aparecen 22800 veces cada uno.
- Solo se seleccionaron PEs FP32.
- No hubo warnings de incompatibilidad de representacion.

## Comando

```bash
run_base=results/objective_sweep_fp32_resnet18_first_layer
.venv/bin/python -u examples/objective_sweep.py \
  --workload workloads/resnet18_first_layer.onnx \
  --ip-pool configs/ip_pool_fp32_65nm.yaml \
  --results-dir "$run_base" \
  --level1-pop-size 80 \
  --level1-generations 8 \
  --workers 16 \
  --level2-strategy exhaustive \
  --level2-exhaustive-max-combinations 100000 \
  --max-architectures 12 \
  --no-constraints \
  --seed 1
```
