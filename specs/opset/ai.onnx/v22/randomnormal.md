---
inherit: v1/randomnormal
since_version: 22
type_constraints:
  update:
  - name: T
    doc: Constrain output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
