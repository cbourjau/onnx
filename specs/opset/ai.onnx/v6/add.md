---
inherit: v1/add
since_version: 6
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to high-precision numeric tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(uint32)
    - tensor(uint64)
attributes:
  remove:
  - consumed_inputs
---
