---
inherit: v1/size
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Input tensor can be of arbitrary type.
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
inputs:
  update:
  - name: data
    type: T
    doc: An input tensor.
    differentiable: false
outputs:
  update:
  - name: size
    type: T1
    doc: Total number of elements of the input tensor
    differentiable: false
---
