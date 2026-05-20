---
inherit: v8/expand
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all tensors.
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
    doc: Input tensor
    differentiable: true
  - name: shape
    type: tensor(int64)
    doc: A 1-D tensor indicates the shape you want to expand to, following the broadcast
      rule
    differentiable: false
outputs:
  update:
  - name: output
    type: T
    doc: Output tensor
    differentiable: true
---
