---
inherit: v9/eyelike
since_version: 22
type_constraints:
  update:
  - name: T1
    doc: Constrain input types. Strings and complex are not supported.
    allowed:
    - tensor(bfloat16)
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
  - name: T2
    doc: Constrain output types. Strings and complex are not supported.
    allowed:
    - tensor(bfloat16)
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
attributes:
  update:
  - name: dtype
    type: int
    doc: (Optional) The data type for the elements of the output tensor. If not specified,
      the data type of the input tensor T1 is used.
    required: false
---
