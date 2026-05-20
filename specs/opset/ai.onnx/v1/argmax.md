---
op: ArgMax
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to all numeric tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: data
  type: T
  doc: An input tensor.
outputs:
- name: reduced
  type: tensor(int64)
  doc: Reduced output tensor with integer data type.
attributes:
- name: axis
  type: int
  doc: The axis in which to compute the arg indices.
  required: false
  default: 0
- name: keepdims
  type: int
  doc: Keep the reduced dimension or not, default 1 means keep reduced dimension.
  required: false
  default: 1
---

Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
The type of the output tensor is integer.
