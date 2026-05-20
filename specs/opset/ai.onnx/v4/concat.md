---
inherit: v1/concat
since_version: 4
type_constraints:
  update:
  - name: T
    doc: Constrain output types to any tensor type.
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
attributes:
  update:
  - name: axis
    type: int
    doc: Which axis to concat on
    required: true
---
