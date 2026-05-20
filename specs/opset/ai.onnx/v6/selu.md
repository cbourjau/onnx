---
inherit: v1/selu
since_version: 6
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  Selu <alpha,gamma>(X) => (Y)
  {
     Alpha = Constant <value_float: float = @alpha> ()
     AlphaCast = CastLike (Alpha, X)
     Gamma = Constant <value_float: float = @gamma> ()
     GammaCast = CastLike (Gamma, X)
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, X)
     ExpX = Exp (X)
     AlphaMulExpX = Mul (AlphaCast, ExpX)
     AlphaMulExpXSubAlpha = Sub (AlphaMulExpX, AlphaCast)
     Neg = Mul (GammaCast, AlphaMulExpXSubAlpha)
     Pos = Mul (GammaCast, X)
     XLessThanZero = Less (X, ZeroCast)
     Y = Where (XLessThanZero, Neg, Pos)
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
    doc: Coefficient of SELU default to 1.67326319217681884765625 (i.e., float32 approximation
      of 1.6732632423543772848170429916717).
    required: false
    default: 1.6732631921768188
  - name: gamma
    type: float
    doc: Coefficient of SELU default to 1.05070102214813232421875 (i.e., float32 approximation
      of 1.0507009873554804934193349852946).
    required: false
    default: 1.0507010221481323
---
