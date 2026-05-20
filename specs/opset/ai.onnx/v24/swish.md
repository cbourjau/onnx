---
op: Swish
domain: ai.onnx
since_version: 24
function_body: |-
  <
    domain: "",
    opset_import: ["" : 24]
  >
  Swish <alpha>(X) => (Y)
  {
     Alpha = Constant <value_float: float = @alpha> ()
     AlphaCast = CastLike (Alpha, X)
     AlphaMulX = Mul (AlphaCast, X)
     SigmoidAlphaMulX = Sigmoid (AlphaMulX)
     Y = Mul (X, SigmoidAlphaMulX)
  }
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(bfloat16)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: X
  type: T
  doc: Input tensor
  differentiable: true
outputs:
- name: Y
  type: T
  doc: Output tensor
  differentiable: true
attributes:
- name: alpha
  type: float
  doc: Coefficient to multiply with input before sigmoid.
  required: false
  default: 1.0
---

Swish function takes one input data (Tensor<T>) and produces one output data (Tensor<T>) of the same shape,
where $Swish(x) = x * sigmoid(alpha * x)$.
