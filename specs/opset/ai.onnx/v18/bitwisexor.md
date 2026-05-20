---
op: BitwiseXor
domain: ai.onnx
since_version: 18
type_constraints:
- name: T
  doc: Constrain input to integer tensors.
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
- name: A
  type: T
  doc: First input operand for the bitwise operator.
  differentiable: false
- name: B
  type: T
  doc: Second input operand for the bitwise operator.
  differentiable: false
outputs:
- name: C
  type: T
  doc: Result tensor.
  differentiable: false
---

Returns the tensor resulting from performing the bitwise `xor` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
