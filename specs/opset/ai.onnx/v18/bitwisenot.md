---
op: BitwiseNot
domain: ai.onnx
since_version: 18
type_constraints:
- name: T
  doc: Constrain input/output to integer tensors.
  allowed:
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
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

Returns the bitwise not of the input tensor element-wise.
