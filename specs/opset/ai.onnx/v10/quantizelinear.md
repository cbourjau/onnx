---
op: QuantizeLinear
domain: ai.onnx
since_version: 10
type_constraints:
- name: T1
  doc: Constrain 'x' to float or int32 tensor.
  allowed:
  - tensor(float)
  - tensor(int32)
- name: T2
  doc: Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
inputs:
- name: x
  type: T1
  doc: N-D full precision Input tensor to be quantized.
- name: y_scale
  type: tensor(float)
  doc: Scale for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer
    quantization.
- name: y_zero_point
  type: T2
  doc: Zero point for doing quantization to get 'y'. It's a scalar, which means a
    per-tensor/layer quantization. Default value is uint8 typed 0 if it's not specified.
  kind: Option
outputs:
- name: y
  type: T2
  doc: N-D quantized output tensor. It has same shape as input 'x'.
---

The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
