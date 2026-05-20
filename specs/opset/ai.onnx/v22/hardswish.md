---
inherit: v14/hardswish
since_version: 22
function_body: |-
  <
    domain: "",
    opset_import: ["" : 22]
  >
  HardSwish (X) => (Y)
  {
     HS_X = HardSigmoid <alpha: float = 0.166667, beta: float = 0.5> (X)
     Y = Mul (X, HS_X)
  }
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
