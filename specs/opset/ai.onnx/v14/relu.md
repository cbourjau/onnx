---
inherit: v13/relu
since_version: 14
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  Relu (X) => (Y)
  {
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, X)
     Y = Max (X, ZeroCast)
  }
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to signed numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
---
