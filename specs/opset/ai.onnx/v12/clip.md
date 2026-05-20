---
inherit: v11/clip
since_version: 12
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all numeric tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
---
