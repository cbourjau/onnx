---
inherit: v23/dequantizelinear
since_version: 24
type_constraints:
  update:
  - name: T2
    doc: The type of the input 'x_scale'.
    allowed:
    - tensor(bfloat16)
    - tensor(float)
    - tensor(float16)
    - tensor(float8e8m0)
---
