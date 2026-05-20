---
inherit: v7/pow
since_version: 12
type_constraints:
  add:
  - name: T1
    doc: Constrain input Y types to float/int tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
  update:
  - name: T
    doc: Constrain input X and output types to float/int tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
inputs:
  update:
  - name: Y
    type: T1
    doc: Second operand, power of the exponent.
---
