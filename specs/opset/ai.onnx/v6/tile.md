---
inherit: v1/tile
since_version: 6
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
  - name: T1
    doc: Constrain repeat's type to int64 tensors.
    allowed:
    - tensor(int64)
inputs:
  add:
  - name: repeats
    type: T1
    doc: 1D int64 tensor of the same length as input's dimension number, includes
      numbers of repeated copies along input's dimensions.
  remove:
  - tiles
  - axis
outputs:
  update:
  - name: output
    type: T
    doc: Output tensor of the same dimensions and type as tensor input. output_dim[i]
      = input_dim[i] * repeats[i]
---

Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
