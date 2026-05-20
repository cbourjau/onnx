---
op: Mish
domain: ai.onnx
since_version: 18
function_body: |-
  <
    domain: "",
    opset_import: ["" : 18]
  >
  Mish (X) => (Y)
  {
     Softplus_X = Softplus (X)
     TanHSoftplusX = Tanh (Softplus_X)
     Y = Mul (X, TanHSoftplusX)
  }
type_constraints:
- name: T
  doc: Constrain input X and output types to float tensors.
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

Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Perform the linear unit element-wise on the input tensor X using formula:

```
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
```
