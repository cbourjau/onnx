---
inherit: v21/constantofshape
since_version: 23
type_constraints:
  update:
  - name: T2
    doc: Constrain output types to be numerics or boolean.
    allowed:
    - tensor(bfloat16)
    - tensor(bool)
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
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint4)
    - tensor(uint64)
    - tensor(uint8)
---
