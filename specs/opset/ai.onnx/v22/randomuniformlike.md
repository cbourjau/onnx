---
inherit: v1/randomuniformlike
since_version: 22
type_constraints:
  update:
  - name: T1
    doc: Constrain to any tensor type. If the dtype attribute is not provided this
      must be a valid output type.
    allowed:
    - tensor(bfloat16)
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
  - name: T2
    doc: Constrain output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
