---
inherit: v1/neg
since_version: 6
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to signed numeric tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
attributes:
  remove:
  - consumed_inputs
---
