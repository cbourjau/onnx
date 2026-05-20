---
op: Unsqueeze
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to all tensor types.
  allowed:
  - tensor(bool)
  - tensor(complex128)
  - tensor(complex64)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(string)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: data
  type: T
  doc: Original tensor
outputs:
- name: expanded
  type: T
  doc: Reshaped tensor with same data as input.
attributes:
- name: axes
  type: ints
  doc: List of non-negative integers, indicate the dimensions to be inserted
  required: true
---

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]
