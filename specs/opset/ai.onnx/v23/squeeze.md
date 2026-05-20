---
inherit: v21/squeeze
since_version: 23
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all tensor types up to IRv11.
    allowed:
    - tensor(bfloat16)
    - tensor(bool)
    - tensor(complex128)
    - tensor(complex64)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(float4e2m1)
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
    - tensor(int16)
    - tensor(int32)
    - tensor(int4)
    - tensor(int64)
    - tensor(int8)
    - tensor(string)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint4)
    - tensor(uint64)
    - tensor(uint8)
inputs:
  update:
  - name: axes
    type: tensor(int64)
    doc: 1D tensor of integers indicating the dimensions to squeeze. Negative value
      means counting dimensions from the back. Accepted range is [-r, r-1] where r
      = rank(data).
    kind: Option
    differentiable: false
---
