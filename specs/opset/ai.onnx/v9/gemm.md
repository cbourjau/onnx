---
inherit: v7/gemm
since_version: 9
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float/int tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(uint32)
    - tensor(uint64)
---
