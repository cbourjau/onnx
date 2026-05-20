---
op: SpaceToDepth
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
- name: input
  type: T
  doc: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth,
    H is the height and W is the width.
outputs:
- name: output
  type: T
  doc: Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].
attributes:
- name: blocksize
  type: int
  doc: Blocks of [blocksize, blocksize] are moved.
  required: true
---

SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
