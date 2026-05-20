---
inherit: v7/multinomial
since_version: 22
type_constraints:
  update:
  - name: T1
    doc: Constrain input types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
