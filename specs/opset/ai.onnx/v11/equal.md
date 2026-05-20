---
inherit: v7/equal
since_version: 11
type_constraints:
  update:
  - name: T
    doc: Constrain input types to all numeric tensors.
    allowed:
    - tensor(bool)
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
