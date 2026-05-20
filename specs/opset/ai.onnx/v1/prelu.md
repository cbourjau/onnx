---
op: PRelu
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
- name: X
  type: T
  doc: Input tensor
- name: slope
  type: T
  doc: Slope tensor. If `Slope` is of size 1, the value is sharedacross different
    channels
outputs:
- name: Y
  type: T
  doc: Output tensor
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
---

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
