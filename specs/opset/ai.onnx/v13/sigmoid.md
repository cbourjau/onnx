---
inherit: v6/sigmoid
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
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
