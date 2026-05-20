---
inherit: v12/pow
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input X and output types to float/int tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
inputs:
  update:
  - name: X
    type: T
    doc: First operand, base of the exponent.
    differentiable: true
  - name: Y
    type: T1
    doc: Second operand, power of the exponent.
    differentiable: true
outputs:
  update:
  - name: Z
    type: T
    doc: Output tensor
    differentiable: true
---
