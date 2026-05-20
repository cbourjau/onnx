---
op: DequantizeLinear
domain: ai.onnx
since_version: 10
type_constraints:
- name: T
  doc: Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.
  allowed:
  - tensor(int32)
  - tensor(int8)
  - tensor(uint8)
inputs:
- name: x
  type: T
  doc: N-D quantized input tensor to be de-quantized.
- name: x_scale
  type: tensor(float)
  doc: Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization.
- name: x_zero_point
  type: T
  doc: Zero point for input 'x'. It's a scalar, which means a per-tensor/layer quantization.
    It's optional. 0 is the default value when it's not specified.
  kind: Option
outputs:
- name: y
  type: tensor(float)
  doc: N-D full precision output tensor. It has same shape as input 'x'.
---

The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' are both scalars.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
