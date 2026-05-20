---
op: Exp
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
  doc: Input tensor
outputs:
- name: output
  type: T
  doc: The exponential of the input tensor computed element-wise
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
---

Calculates the exponential of the given input tensor, element-wise.
