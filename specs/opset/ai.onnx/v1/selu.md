---
op: Selu
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
  doc: Coefficient of SELU default to 1.6732.
  required: false
  default: 1.673200011253357
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
- name: gamma
  type: float
  doc: Coefficient of SELU default to 1.0507.
  required: false
  default: 1.0506999492645264
---

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
