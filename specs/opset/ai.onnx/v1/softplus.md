---
op: Softplus
domain: ai.onnx
since_version: 1
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  Softplus (X) => (Y)
  {
     exp_x = Exp (X)
     one = Constant <value: tensor = float {1}> ()
     one_cast = CastLike (one, X)
     exp_x_add_one = Add (exp_x, one_cast)
     Y = Log (exp_x_add_one)
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
---

Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
