---
inherit: v11/cumsum
since_version: 14
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(uint32)
    - tensor(uint64)
---
