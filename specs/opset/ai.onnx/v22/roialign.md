---
inherit: v16/roialign
since_version: 22
type_constraints:
  update:
  - name: T1
    doc: Constrain types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
