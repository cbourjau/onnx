---
op: Resize
domain: ai.onnx
since_version: 10
type_constraints:
- name: T
  doc: Constrain input 'X' and output 'Y' to all tensor types.
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
- name: X
  type: T
  doc: N-D tensor
- name: scales
  type: tensor(float)
  doc: The scale array along each dimension. It takes value greater than 0. If it's
    less than 1, it's sampling down, otherwise, it's upsampling. The number of elements
    of 'scales' should be the same as the rank of input 'X'.
outputs:
- name: Y
  type: T
  doc: N-D tensor after resizing
attributes:
- name: mode
  type: string
  doc: 'Two interpolation modes: nearest (default), and linear (including bilinear,
    trilinear, etc)'
  required: false
  default: nearest
---

Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
