# Comparación con aceleradores académicos de inferencia

Este documento recopila procesadores y aceleradores académicos fabricados y medidos que publican cifras de área, frecuencia, rendimiento y energía por inferencia. Sirve como referencia para contextualizar los IP caracterizados en [`talos_ip_pool_tsmc65.yaml`](./talos_ip_pool_tsmc65.yaml).

El procesador de KU Leuven al que probablemente se hacía referencia es **DIANA**. Para una comparación digital más cercana a los PE INT8 caracterizados en TSMC 65 nm, las referencias más útiles son **TinyVers**, **Eyeriss v1** y **UNPU**.

## Resultados publicados

| Precisión del resultado interno | Precisión de salida | Procesador | Tecnología | Área publicada | Frecuencia | Red y precisión | Precisión del PE / datapath | Rendimiento medido | Energía por inferencia | Alcance de la energía |
|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| INT11[^result-cutie] | TERN2 (`−1, 0, +1`) | [TCN-CUTIE / Kraken](https://arxiv.org/pdf/2212.00688) | GF 22FDX | 2.96 mm² acelerador; 9 mm² chip | 54 MHz | CIFAR-10, CNN ternaria W/A | Ternaria W/A (`−1, 0, +1`) | 3,200 inf/s; 5.4 TOPS medios | **2.72 µJ** | Acelerador y memorias internas; excluye I/O del chip |
| Analógica SC; sin ancho digital | BIN1 | [Mixed-signal BNN processor](https://doi.org/10.1109/JSSC.2018.2869150) | 28 nm | 6.0 mm² chip | ≈10 MHz[^frequency-bnn] | CIFAR-10, BNN | W1/A1 binaria (`−1, +1`); suma analógica | 237 inf/s | **3.8 µJ** | Clasificación completa imagen→etiqueta; memoria on-chip |
| AIMC: analógica → ADC6; digital: N/D | AIMC: INT6; digital: INT8 | [DIANA](https://imec-publications.be/entities/publication/5a4187a7-0240-4099-8500-5996317b4480/full) | GF 22FDX | 8.91 mm² sin pads; 10.24 mm² chip; AIMC 2.29 mm² | 250 MHz nominal; funcional hasta 270 MHz | ResNet-20/CIFAR-10, digital + analógico | AIMC: I7/W1.5/O6; digital: I8/W8/O8 | ≈794 inf/s; 1.26 ms | **≈11.6 µJ**[^energy-diana] | Inferencia end-to-end; primera capa digital y resto principalmente AIMC |
| UINT11 popcount[^result-binareye] | BIN1 | [BinarEye](https://arxiv.org/pdf/1804.05554) | 28 nm | 1.4 mm² activa; ≈2 mm² chip | 1.5–48 MHz | CIFAR-10, BNN de 9 capas | W1/A1 binaria; entrada RGB de 7 bit | 150 inf/s | **14.4 µJ** | Inferencia completa, incluyendo memorias on-chip |
| INT32 | INT8/4/2 | [TinyVers](https://arxiv.org/pdf/2301.03537) | GF 22FDX | 6.25 mm² chip | 5 MHz en esta medición; máximo 150 MHz | ResNet-8/CIFAR-10, INT8 | INT8/4/2; acumulación INT32 | 13.2 inf/s; 76 ms; pico HW 17.6 GOPS | **17.3 µJ**[^energy-tinyvers] | RISC-V + transferencia L2→L1 + acelerador + postproceso; entrada ya en L2 |
| INT10–20, según la capa | INT4 | [NeuroCorgi](https://list.cea.fr/app/uploads/2024/11/NeuroCorgi-APCCAS.pdf) | GF 22FDX | 4.45 mm² FEA; 7.86 mm² chip | 59 MHz | MobileNet-v1, ImageNet, W4/A4 | INT4 W/A | 788 imágenes/s; ≈1.23 ms | **36.7 µJ** | Sólo el extractor de características fijo; no incluye el clasificador FC final |
| N/D | INT16 | [UNPU](https://doi.org/10.1109/ISSCC.2018.8310262) | 65 nm | 16 mm² chip | 5–200 MHz | AlexNet, convoluciones con pesos INT8 | A16 × W1–16, punto fijo bit-serial | 346 imágenes/s | **≈911 µJ**[^energy-unpu] | Capas convolucionales; valor derivado de 1,097.5 imágenes/J |
| INT12/24/48 para SIMD 4×/2×/1× | INT1–16 | [Envision](https://lirias.kuleuven.be/bitstream/123456789/579133/1/isscc_oa.pdf) | 28 nm FDSOI | 1.87 mm² activa | 200 MHz a 1 V | AlexNet, convoluciones, precisión escalable | W/A de punto fijo 1–16 bit; SIMD 4/8/16 bit | 47 imágenes/s | **≈936 µJ**[^energy-envision] | Sólo convoluciones; normalmente no incluye DRAM externa |
| INT16 pSum | INT16 | [Eyeriss v1](https://dspace.mit.edu/bitstream/handle/1721.1/101151/eyeriss_isscc_2016.pdf) | TSMC 65 nm LP | 12.25 mm² core | 200 MHz core; 60 MHz enlace | AlexNet, convoluciones INT16, batch 4 | W16/A16, punto fijo | 34.7 imágenes/s; pico 33.6 GMAC/s | **≈8,010 µJ**[^energy-eyeriss] | Potencia del chip; excluye la energía de DRAM externa |

El resultado interno es la suma o acumulación anterior a activación y requantización; la salida es el dato entregado a la siguiente capa. `W`, `A`, `I` y `O` indican pesos, activaciones, entrada y salida, respectivamente; el número indica el ancho en bits. `N/D` significa que el artículo no publica el ancho.

[^result-cutie]: Ancho mínimo derivado para la suma con `3 × 3 × 96 = 864` productos ternarios; la arquitectura no escribe sumas parciales a memoria.
[^result-binareye]: Ancho mínimo derivado para el popcount de `2 × 2 × 256 = 1,024` resultados XNOR.
[^frequency-bnn]: Frecuencia derivada de `237 fps / 23.8 fps/MHz ≈ 10 MHz`.
[^energy-diana]: Aproximación calculada como `81.4 MOp / 7.02 TOPS/W ≈ 11.6 µJ`. Debe tratarse como aproximación porque el artículo comunica principalmente la eficiencia end-to-end en TOPS/W.
[^energy-tinyvers]: Calculada a partir de la potencia y latencia publicadas: `228 µW × 76 ms ≈ 17.3 µJ`.
[^energy-unpu]: Calculada a partir de la eficiencia publicada: `1 / 1,097.5 imágenes/J ≈ 911 µJ/imagen`.
[^energy-envision]: Calculada a partir de aproximadamente 44 mW y 47 imágenes/s: `44 mW / 47 imágenes/s ≈ 936 µJ/imagen`.
[^energy-eyeriss]: Calculada a partir de 278 mW y 34.7 imágenes/s: `278 mW / 34.7 imágenes/s ≈ 8.01 mJ/imagen`.

## Comparación con los PE caracterizados

Las siguientes cifras proceden del pool TSMC 65 nm local. El área original del YAML está expresada en µm²; aquí se convierte a mm² dividiendo entre un millón.

| Precisión del resultado interno | Precisión de salida | IP local | Precisión de operandos | Área | Fmax | Energía dinámica |
|---|---|---|---|---:|---:|---:|
| INT32 | INT32 | `pe_simple_mac` | INT8 × INT8 | 0.001166 mm² | 268.5 MHz | 1.744 pJ/MAC |
| UINT32 | UINT32 | `pe_tms4517_simple` | UINT8 × UINT8 | 0.001275 mm² | 322.3 MHz | 1.649 pJ/MAC |
| INT32 | INT32 | `pe_tiny_xpu` | INT8 × INT8 | 0.001293 mm² | 267.4 MHz | 1.782 pJ/MAC |
| INT32 | INT32 | `pe_gemmini_like` | INT8 × INT8 | 0.002025 mm² | 253.7 MHz | **1.449 pJ/MAC** |
| INT32 | INT32 | `pe_sauria` | INT8 × INT8 | 0.002398 mm² | 287.6 MHz | 1.978 pJ/MAC |
| INT32 | INT32 | `pe_alibaba_int8` | INT8 × INT8 | 0.002676 mm² | 346.9 MHz | 3.120 pJ/MAC |
| 2 × INT32 | 2 × INT32 (64 bit) | `pe_openeye_composite_int8` | INT8 × INT8; 2 MAC/ciclo; RF incluidos | 0.070829 mm² | 278.6 MHz | 0.216 pJ/MAC[^openeye-energy] |

En estos wrappers locales no se modela una requantización posterior: la salida conserva la precisión del acumulador. OpenEye agrupa dos resultados INT32 en su RF de salida de 64 bit.

[^openeye-energy]: Es energía dinámica incremental respecto a un elevado baseline idle y no es directamente equivalente a la energía total de una PE convencional. El bloque contiene datapath compuesto y scratchpads/RF locales.

## Interpretación

Los IP locales son **PE individuales sintetizados**, mientras que los trabajos académicos de la primera tabla son aceleradores o SoC completos. Estos últimos pueden incluir arrays de PE, SRAM, interconexión, control, CPU, buffers y pads. Por tanto, no se debe comparar directamente el área de una PE local con el área total de TinyVers, DIANA, UNPU o Eyeriss.

Las referencias pueden ordenarse por utilidad para esta comparación:

1. **Eyeriss v1 y UNPU:** referencias de proceso, ya que también emplean 65 nm.
2. **TinyVers:** referencia digital INT8 con una medición end-to-end relativamente bien delimitada.
3. **DIANA:** referencia de eficiencia híbrida digital/analógica; no constituye una comparación directa con lógica digital estándar.
4. **BinarEye y CUTIE:** límites especializados con pesos y activaciones binarios o ternarios.
5. **NeuroCorgi:** referencia para MobileNet, aunque mide un extractor fijo y no la clasificación completa.

También hay diferencias importantes entre workloads. Una inferencia de CIFAR-10 no tiene el mismo coste que una inferencia de ImageNet, y una red binaria o ternaria no es equivalente a una red INT8 o INT16. La comparación debe conservar siempre las columnas de red, dataset, precisión y alcance energético.

## Métrica necesaria para una comparación directa

Para producir una fila local comparable con la literatura sería necesario instanciar un array completo con sus memorias, mapear una red concreta y medir:

- área total del acelerador o SoC en mm²;
- frecuencia efectiva del benchmark;
- latencia e inferencias por segundo con `batch = 1`, salvo que se declare otro batch;
- potencia media durante toda la inferencia;
- energía por inferencia;
- precisión numérica, red, dataset y exactitud del modelo;
- inclusión o exclusión de SRAM, DRAM, CPU, I/O y potencia idle.

La energía de inferencia se calcula como:

$$
E_{\text{inferencia}} = \int_0^{t_{\text{inferencia}}} P(t)\,dt
\approx P_{\text{media}}\,t_{\text{inferencia}}.
$$

El campo `dynamic_energy_per_mac_pj` del YAML representa la energía dinámica incremental de una operación MAC bajo caracterización vectorless. No incluye por sí solo SRAM, movimiento de datos, control, árbol de reloj, períodos ociosos ni el resto de la plataforma, por lo que no debe presentarse como energía por inferencia.

## Recomendación de benchmark

La comparación más defendible con los datos existentes sería construir un acelerador INT8 completo y ejecutar **ResNet-8 sobre CIFAR-10**, reproduciendo en lo posible el benchmark de TinyVers. Esto permitiría reportar una fila homogénea con área total, frecuencia, latencia, inferencias/s y energía/inferencia. Como segunda referencia, un benchmark de convoluciones AlexNet permitiría contrastar con Eyeriss y UNPU en el mismo nodo de 65 nm, aunque su alcance no sería end-to-end.
