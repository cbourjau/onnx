---
op: Constant
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
outputs:
- name: output
  type: T
  doc: Output tensor containing the same value of the provided tensor.
attributes:
- name: value
  type: tensor
  doc: The value for the elements of the output tensor.
  required: true
---

A constant tensor.
