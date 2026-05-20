---
inherit: v9/meanvariancenormalization
since_version: 13
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  MeanVarianceNormalization <axes>(X) => (Y)
  {
     Exponent = Constant <value: tensor = float {2}> ()
     Epsilon = Constant <value: tensor = float {1e-09}> ()
     axes = Constant <value_ints: ints = @axes> ()
     X_RM = ReduceMean (X, axes)
     EX_squared = Pow (X_RM, Exponent)
     X_squared = Pow (X, Exponent)
     E_Xsquared = ReduceMean (X_squared, axes)
     Variance = Sub (E_Xsquared, EX_squared)
     STD = Sqrt (Variance)
     X_variance = Sub (X, X_RM)
     Processed_STD = Add (STD, Epsilon)
     Y = Div (X_variance, Processed_STD)
  }
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
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
---

A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
