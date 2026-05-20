---
inherit: v1/upsample
since_version: 7
support_level: common
type_constraints:
  update:
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
  update:
  - name: X
    type: T
    doc: N-D tensor
outputs:
  update:
  - name: Y
    type: T
    doc: N-D tensor after resizing
attributes:
  add:
  - name: scales
    type: floats
    doc: The scale array along each dimension. It takes value greater than or equal
      to 1. The number of elements of 'scales' should be the same as the rank of input
      'X'.
    required: true
  remove:
  - height_scale
  - width_scale
  update:
  - name: mode
    type: string
    doc: 'Two interpolation modes: nearest (default), and linear (including bilinear,
      trilinear, etc)'
    required: false
    default: nearest
---

Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
