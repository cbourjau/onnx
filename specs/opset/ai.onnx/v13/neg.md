---
inherit: v6/neg
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to signed numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
inputs:
  update:
  - name: X
    type: T
    doc: Input tensor
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor
    differentiable: true
---
