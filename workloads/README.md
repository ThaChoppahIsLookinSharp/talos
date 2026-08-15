# ONNX workloads

The bundled files are lightweight research workloads for Talos and
ZigZag.

| File | Source | Convolution format |
|---|---|---|
| `alexnet_int16.onnx` | Eyeriss / BVLC AlexNet | INT16 / INT16 / INT16 |
| `squeezenet1_0_int8.onnx` | ONNX Model Zoo | UINT8 / INT8 / UINT8 |
| `squeezenet1_0_fp16.onnx` | WebNN | FP16 / FP16 / FP16 |

The format columns correspond to activation, weight, and output. The
INT8 model follows the common asymmetric convention: its public input
and output are FP32, while each `QLinearConv` uses UINT8 activations,
INT8 weights, and UINT8 outputs. Talos infers those internal types.

`alexnet_int16.onnx` contains the canonical AlexNet topology and the
227-pixel input used by Eyeriss, represented as signed 16-bit
fixed-point. Eyeriss did not publish its quantized weights or per-layer
scales, so this is intentionally a shape-only workload, not an
executable inference checkpoint. It retains all five convolutional and
three fully-connected layers; the Eyeriss chip measurements cover the
five convolutional layers.

Sources:

- [BVLC AlexNet][alexnet-source], the Caffe model linked by Eyeriss.
- [`nn_dataflow` AlexNet][nn-dataflow-source], which reproduces the
  Eyeriss ISSCC/JSSC layer geometry and defaults to a 16-bit word.
- [Eyeriss benchmark data][eyeriss-source], reporting 16-bit weights
  and input activations for AlexNet.
- [SqueezeNet 1.0 INT8][int8-source], quantized with Intel Neural
  Compressor and published by the ONNX Model Zoo.
- [SqueezeNet 1.0 FP16][fp16-source], published by the WebNN project.

The two SqueezeNet sources declare those models under Apache-2.0.

SHA-256:

```text
f074a2fe745884e9fcd3cbf62c45f94a7a18b389d2ca19acb9f031db92fe62f4
  alexnet_int16.onnx
3da17dfad1b7ba23c93fac6dbf49f6db78cd42f7519e915a2e27d37c5c0a972b
  squeezenet1_0_int8.onnx
ec43aca36ab96cc578af9d724bf58d703db0d0071ef4f4310a05327a9845ab18
  squeezenet1_0_fp16.onnx
```

[alexnet-source]: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt
[nn-dataflow-source]: https://github.com/stanford-mast/nn_dataflow/blob/198a5274b9529125c6aa2b8b72b365d60cf83778/nn_dataflow/nns/alex_net.py
[eyeriss-source]: https://eyeriss.mit.edu/benchmarking.html
[int8-source]: https://huggingface.co/onnxmodelzoo
[fp16-source]: https://huggingface.co/webnn
