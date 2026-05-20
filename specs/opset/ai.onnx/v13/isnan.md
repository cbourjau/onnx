---
inherit: v9/isnan
since_version: 13
type_constraints:
  update:
  - name: T1
    doc: Constrain input types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
inputs:
  update:
  - name: X
    type: T1
    doc: input
    differentiable: false
outputs:
  update:
  - name: Y
    type: T2
    doc: output
    differentiable: false
---
