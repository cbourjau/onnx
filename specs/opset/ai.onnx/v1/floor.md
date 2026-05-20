---
op: Floor
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
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
---

Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
