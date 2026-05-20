---
op: Size
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Input tensor can be of arbitrary type.
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
- name: T1
  doc: Constrain output to int64 tensor, which should be a scalar though.
  allowed:
  - tensor(int64)
inputs:
- name: data
  type: T
  doc: An input tensor.
outputs:
- name: size
  type: T1
  doc: Total number of elements of the input tensor
---

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
