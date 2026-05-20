---
inherit: v1/elu
since_version: 6
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  Elu <alpha>(X) => (Y)
  {
     Alpha = Constant <value_float: float = @alpha> ()
     AlphaCast = CastLike (Alpha, X)
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, X)
     One = Constant <value: tensor = float {1}> ()
     OneCast = CastLike (One, X)
     XLessThanZero = Less (X, ZeroCast)
     ExpX = Exp (X)
     ExpXSubOne = Sub (ExpX, OneCast)
     AlphaMulExpXSubOne = Mul (AlphaCast, ExpXSubOne)
     Y = Where (XLessThanZero, AlphaMulExpXSubOne, X)
  }
inputs:
  update:
  - name: X
    type: T
    doc: Input tensor
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor
    differentiable: true
attributes:
  remove:
  - consumed_inputs
  update:
  - name: alpha
    type: float
    doc: Coefficient of ELU.
    required: false
    default: 1.0
---
