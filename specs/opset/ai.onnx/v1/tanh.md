---
op: Tanh
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
- name: input
  type: T
  doc: 1-D input tensor
outputs:
- name: output
  type: T
  doc: The hyperbolic tangent values of the input tensor computed element-wise
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
---

Calculates the hyperbolic tangent of the given input tensor element-wise.
