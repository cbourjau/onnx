---
inherit: v24/cast
since_version: 25
type_constraints:
  update:
  - name: T1
    doc: Constrain input types. Casting from complex is not supported.
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
    - tensor(float8e8m0)
    - tensor(int16)
    - tensor(int2)
    - tensor(int32)
    - tensor(int4)
    - tensor(int64)
    - tensor(int8)
    - tensor(string)
    - tensor(uint16)
    - tensor(uint2)
    - tensor(uint32)
    - tensor(uint4)
    - tensor(uint64)
    - tensor(uint8)
  - name: T2
    doc: Constrain output types. Casting to complex is not supported.
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
    - tensor(float8e8m0)
    - tensor(int16)
    - tensor(int2)
    - tensor(int32)
    - tensor(int4)
    - tensor(int64)
    - tensor(int8)
    - tensor(string)
    - tensor(uint16)
    - tensor(uint2)
    - tensor(uint32)
    - tensor(uint4)
    - tensor(uint64)
    - tensor(uint8)
---
