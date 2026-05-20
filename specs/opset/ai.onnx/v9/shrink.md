---
op: Shrink
domain: ai.onnx
since_version: 9
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  Shrink <lambd,bias>(input) => (output)
  {
     Lambd = Constant <value_float: float = @lambd> ()
     LambdCast = CastLike (Lambd, input)
     Bias = Constant <value_float: float = @bias> ()
     BiasCast = CastLike (Bias, input)
     Zero = Constant <value: tensor = float {0}> ()
     ZeroCast = CastLike (Zero, input)
     NegLmbda = Neg (LambdCast)
     InputLessThanNegLambda = Less (input, NegLmbda)
     InputAddBias = Add (input, BiasCast)
     InputSubBias = Sub (input, BiasCast)
     LambdaLessThanInput = Less (LambdCast, input)
     InputSubBiasOrZero = Where (LambdaLessThanInput, InputSubBias, ZeroCast)
     output = Where (InputLessThanNegLambda, InputAddBias, InputSubBiasOrZero)
  }
type_constraints:
- name: T
  doc: Constrain input to only numeric types.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: input
  type: T
  doc: The input data as Tensor.
  differentiable: true
outputs:
- name: output
  type: T
  doc: The output.
  differentiable: true
attributes:
- name: bias
  type: float
  doc: The bias value added to output. Default is 0.
  required: false
  default: 0.0
- name: lambd
  type: float
  doc: The lambd value for the Shrink formulation. Default is 0.5.
  required: false
  default: 0.5
---

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.
