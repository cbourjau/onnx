---
inherit: v11/splittosequence
since_version: 24
type_constraints:
  update:
  - name: T
    doc: Constrain input types to all tensor types.
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
  - name: S
    doc: Constrain output types to all tensor types.
    allowed:
    - seq(tensor(bfloat16))
    - seq(tensor(bool))
    - seq(tensor(complex128))
    - seq(tensor(complex64))
    - seq(tensor(double))
    - seq(tensor(float))
    - seq(tensor(float16))
    - seq(tensor(int16))
    - seq(tensor(int32))
    - seq(tensor(int64))
    - seq(tensor(int8))
    - seq(tensor(string))
    - seq(tensor(uint16))
    - seq(tensor(uint32))
    - seq(tensor(uint64))
    - seq(tensor(uint8))
---
