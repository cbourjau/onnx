---
inherit: v1/hardsigmoid
since_version: 6
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  HardSigmoid <alpha,beta>(X) => (Y)
  {
     Alpha = Constant <value_float: float = @alpha> ()
     AlphaCast = CastLike (Alpha, X)
     Beta = Constant <value_float: float = @beta> ()
     BetaCast = CastLike (Beta, X)
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, X)
     One = Constant <value: tensor = float {1}> ()
     OneCast = CastLike (One, X)
     AlphaMulX = Mul (X, AlphaCast)
     AlphaMulXAddBeta = Add (AlphaMulX, BetaCast)
     MinOneOrAlphaMulXAddBeta = Min (AlphaMulXAddBeta, OneCast)
     Y = Max (MinOneOrAlphaMulXAddBeta, ZeroCast)
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
    doc: Value of alpha.
    required: false
    default: 0.20000000298023224
  - name: beta
    type: float
    doc: Value of beta.
    required: false
    default: 0.5
---
