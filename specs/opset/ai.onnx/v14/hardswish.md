---
op: HardSwish
domain: ai.onnx
since_version: 14
function_body: |-
  <
    domain: "",
    opset_import: ["" : 14]
  >
  HardSwish (X) => (Y)
  {
     HS_X = HardSigmoid <alpha: float = 0.166667, beta: float = 0.5> (X)
     Y = Mul (X, HS_X)
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

HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
