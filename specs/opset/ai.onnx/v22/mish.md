---
inherit: v18/mish
since_version: 22
function_body: |-
  <
    domain: "",
    opset_import: ["" : 22]
  >
  Mish (X) => (Y)
  {
     Softplus_X = Softplus (X)
     TanHSoftplusX = Tanh (Softplus_X)
     Y = Mul (X, TanHSoftplusX)
  }
type_constraints:
  update:
  - name: T
    doc: Constrain input X and output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
