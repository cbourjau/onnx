---
inherit: v6/tile
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all tensor types.
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
  - name: input
    type: T
    doc: Input tensor of any shape.
    differentiable: true
  - name: repeats
    type: T1
    doc: 1D int64 tensor of the same length as input's dimension number, includes
      numbers of repeated copies along input's dimensions.
    differentiable: false
outputs:
  update:
  - name: output
    type: T
    doc: Output tensor of the same dimensions and type as tensor input. output_dim[i]
      = input_dim[i] * repeats[i]
    differentiable: true
---
