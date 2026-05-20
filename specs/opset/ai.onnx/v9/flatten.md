---
inherit: v1/flatten
since_version: 9
type_constraints:
  update:
  - name: T
    doc: Constrain input and output to all tensor types.
    allowed:
    - tensor(bool)
    - tensor(complex128)
    - tensor(complex64)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(string)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
---
