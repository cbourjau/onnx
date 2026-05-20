---
op: Min
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
- name: data_0
  type: T
  doc: List of tensors for Min
  kind:
    homogeneous: true
    min_arity: 1
outputs:
- name: min
  type: T
  doc: Output tensor. Same dimension as inputs.
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
---

Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
