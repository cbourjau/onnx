---
op: Elu
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
outputs:
- name: Y
  type: T
  doc: Output tensor
attributes:
- name: alpha
  type: float
  doc: Coefficient of ELU default to 1.0.
  required: false
  default: 1.0
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
---

Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
