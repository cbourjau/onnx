---
inherit: v23/castlike
since_version: 24
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
attributes:
  add:
  - name: round_mode
    type: string
    doc: 'Rounding mode for conversion to float8e8m0. It only applies to casting to
      float8e8m0 and is `up` by default. `up`: round to nearest value away from zero,
      `down`: round to nearest value towards zero, `nearest`: round to nearest value
      and ties round up. Please refer to operator Cast description for further details.'
    required: false
    default: up
  update:
  - name: saturate
    type: int
    doc: The parameter defines how the conversion behaves if an input value is out
      of range of the destination type. It only applies for float 8 conversion (float8e4m3fn,
      float8e4m3fnuz, float8e5m2, float8e5m2fnuz, float8e8m0). It is true by default.
      Please refer to operator Cast description for further details.
    required: false
    default: 1
---
