---
op: MatMul
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: A
  type: T
  doc: N-dimensional matrix A
- name: B
  type: T
  doc: N-dimensional matrix B
outputs:
- name: Y
  type: T
  doc: Matrix multiply results from A * B
---

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
