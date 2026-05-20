---
inherit: v11/scatterelements
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Input and output types can be of any tensor type.
    allowed:
    - tensor(bfloat16)
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
  - name: data
    type: T
    doc: Tensor of rank r >= 1.
    differentiable: true
  - name: indices
    type: Tind
    doc: Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index
      values are expected to be within bounds [-s, s-1] along axis of size s. It is
      an error if any of the index values are out of bounds.
    differentiable: false
  - name: updates
    type: T
    doc: Tensor of rank r >=1 (same rank and shape as indices)
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: Tensor of rank r >= 1 (same rank as input).
    differentiable: true
---
