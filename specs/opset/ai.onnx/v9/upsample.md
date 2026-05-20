---
inherit: v7/upsample
since_version: 9
type_constraints:
  update:
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
  add:
  - name: scales
    type: tensor(float)
    doc: The scale array along each dimension. It takes value greater than or equal
      to 1. The number of elements of 'scales' should be the same as the rank of input
      'X'.
attributes:
  remove:
  - scales
---
