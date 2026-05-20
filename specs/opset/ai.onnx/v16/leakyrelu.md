---
inherit: v6/leakyrelu
since_version: 16
function_body: |-
  <
    domain: "",
    opset_import: ["" : 16]
  >
  LeakyRelu <alpha>(X) => (Y)
  {
     Alpha = Constant <value_float: float = @alpha> ()
     AlphaCast = CastLike (Alpha, X)
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, X)
     XLessThanZero = Less (X, ZeroCast)
     AlphaMulX = Mul (AlphaCast, X)
     Y = Where (XLessThanZero, AlphaMulX, X)
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
