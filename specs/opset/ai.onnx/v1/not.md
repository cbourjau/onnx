---
op: Not
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input/output to boolean tensors.
  allowed:
  - tensor(bool)
inputs:
- name: X
  type: T
  doc: Input tensor
  differentiable: false
outputs:
- name: Y
  type: T
  doc: Output tensor
  differentiable: false
---

Returns the negation of the input tensor element-wise.
