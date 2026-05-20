---
inherit: v23/quantizelinear
since_version: 24
type_constraints:
  update:
  - name: T2
    doc: The type of the input 'y_scale'.
    allowed:
    - tensor(bfloat16)
    - tensor(float)
    - tensor(float16)
    - tensor(float8e8m0)
    - tensor(int32)
inputs:
  update:
  - name: y_zero_point
    type: T3
    doc: Zero point for doing quantization to get `y`. Shape must match `y_scale`.
      Default is uint8 with zero point of 0 if it's not specified.
    kind: Option
---
