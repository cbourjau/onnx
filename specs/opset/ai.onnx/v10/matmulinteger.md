---
op: MatMulInteger
domain: ai.onnx
since_version: 10
type_constraints:
- name: T1
  doc: Constrain input A data type to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
- name: T2
  doc: Constrain input B data type to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
- name: T3
  doc: Constrain output Y data type as 32-bit integer tensor.
  allowed:
  - tensor(int32)
inputs:
- name: A
  type: T1
  doc: N-dimensional matrix A
  differentiable: false
- name: B
  type: T2
  doc: N-dimensional matrix B
  differentiable: false
- name: a_zero_point
  type: T1
  doc: 'Zero point tensor for input ''A''. It''s optional and default value is 0.
    It could be a scalar or N-D tensor. Scalar refers to per tensor quantization whereas
    N-D refers to per row quantization. If the input is 2D of shape [M, K] then zero
    point tensor may be an M element vector [zp_1, zp_2, ..., zp_M]. If the input
    is N-D tensor with shape [D1, D2, M, K] then zero point tensor may have shape
    [D1, D2, M, 1]. '
  kind: Option
  differentiable: false
- name: b_zero_point
  type: T2
  doc: 'Zero point tensor for input ''B''. It''s optional and default value is 0.
    It could be a scalar or a N-D tensor, Scalar refers to per tensor quantization
    whereas N-D refers to per col quantization. If the input is 2D of shape [K, N]
    then zero point tensor may be an N element vector [zp_1, zp_2, ..., zp_N]. If
    the input is N-D tensor with shape [D1, D2, K, N] then zero point tensor may have
    shape [D1, D2, 1, N]. '
  kind: Option
  differentiable: false
outputs:
- name: Y
  type: T3
  doc: Matrix multiply results from A * B
  differentiable: false
---

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
