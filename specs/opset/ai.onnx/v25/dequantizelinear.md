---
inherit: v24/dequantizelinear
since_version: 25
type_constraints:
  update:
  - name: T1
    doc: The type of the inputs 'x_zero_point' and 'x'.
    allowed:
    - tensor(float4e2m1)
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
    - tensor(int16)
    - tensor(int2)
    - tensor(int32)
    - tensor(int4)
    - tensor(int8)
    - tensor(uint16)
    - tensor(uint2)
    - tensor(uint4)
    - tensor(uint8)
---
