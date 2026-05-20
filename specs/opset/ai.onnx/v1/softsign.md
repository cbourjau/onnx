---
op: Softsign
domain: ai.onnx
since_version: 1
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  Softsign (input) => (output)
  {
     One = Constant <value: tensor = float {1}> ()
     OneCast = CastLike (One, input)
     AbsInput = Abs (input)
     OneAddAbsInput = Add (OneCast, AbsInput)
     output = Div (input, OneAddAbsInput)
  }
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: input
  type: T
  doc: Input tensor
  differentiable: true
outputs:
- name: output
  type: T
  doc: The softsign (x/(1+|x|)) values of the input tensor computed element-wise
  differentiable: true
---

Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
