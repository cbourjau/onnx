---
inherit: v10/qlinearmatmul
since_version: 21
type_constraints:
  add:
  - name: TS
    doc: Constrain scales.
    allowed:
    - tensor(bfloat16)
    - tensor(float)
    - tensor(float16)
  update:
  - name: T1
    doc: The type of input a and its zeropoint.
    allowed:
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
    - tensor(int8)
    - tensor(uint8)
  - name: T2
    doc: The type of input b and its zeropoint.
    allowed:
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
    - tensor(int8)
    - tensor(uint8)
  - name: T3
    doc: The type of the output and its zeropoint.
    allowed:
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
    - tensor(int8)
    - tensor(uint8)
inputs:
  update:
  - name: a_scale
    type: TS
    doc: scale of quantized input a
    differentiable: false
  - name: b_scale
    type: TS
    doc: scale of quantized input b
    differentiable: false
  - name: y_scale
    type: TS
    doc: scale of quantized output y
    differentiable: false
---
