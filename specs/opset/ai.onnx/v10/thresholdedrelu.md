---
op: ThresholdedRelu
domain: ai.onnx
since_version: 10
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  ThresholdedRelu <alpha>(X) => (Y)
  {
     Alpha = Constant <value_float: float = @alpha> ()
     AlphaCast = CastLike (Alpha, X)
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, X)
     AlphaLessThanX = Less (AlphaCast, X)
     Y = Where (AlphaLessThanX, X, ZeroCast)
  }
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
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
  doc: Threshold value
  required: false
  default: 1.0
---

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
