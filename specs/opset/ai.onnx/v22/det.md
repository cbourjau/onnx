---
inherit: v11/det
since_version: 22
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to floating-point tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
