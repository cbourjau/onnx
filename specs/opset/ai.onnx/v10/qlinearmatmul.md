---
op: QLinearMatMul
domain: ai.onnx
since_version: 10
type_constraints:
- name: T1
  doc: Constrain input a and its zero point data type to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
- name: T2
  doc: Constrain input b and its zero point data type to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
- name: T3
  doc: Constrain output y and its zero point data type to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
inputs:
- name: a
  type: T1
  doc: N-dimensional quantized matrix a
  differentiable: false
- name: a_scale
  type: tensor(float)
  doc: scale of quantized input a
  differentiable: false
- name: a_zero_point
  type: T1
  doc: zero point of quantized input a
  differentiable: false
- name: b
  type: T2
  doc: N-dimensional quantized matrix b
  differentiable: false
- name: b_scale
  type: tensor(float)
  doc: scale of quantized input b
  differentiable: false
- name: b_zero_point
  type: T2
  doc: zero point of quantized input b
  differentiable: false
- name: y_scale
  type: tensor(float)
  doc: scale of quantized output y
  differentiable: false
- name: y_zero_point
  type: T3
  doc: zero point of quantized output y
  differentiable: false
outputs:
- name: y
  type: T3
  doc: Quantized matrix multiply results from a * b
  differentiable: false
---

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
