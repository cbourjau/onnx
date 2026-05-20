---
inherit: v9/nonzero
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain to all tensor types.
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
  - name: X
    type: T
    doc: input
    differentiable: false
outputs:
  update:
  - name: Y
    type: tensor(int64)
    doc: output
    differentiable: false
---
