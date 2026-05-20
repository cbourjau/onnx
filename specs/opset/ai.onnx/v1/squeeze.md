---
op: Squeeze
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
  doc: Tensors with at least max(dims) dimensions.
outputs:
- name: squeezed
  type: T
  doc: Reshaped tensor with same data as input.
attributes:
- name: axes
  type: ints
  doc: List of non-negative integers, indicate the dimensions to squeeze.
  required: false
---

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
