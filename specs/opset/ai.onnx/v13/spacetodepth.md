---
inherit: v1/spacetodepth
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
    doc: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or
      depth, H is the height and W is the width.
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].
    differentiable: true
---
