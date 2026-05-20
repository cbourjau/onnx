---
op: Gelu
domain: ai.onnx
since_version: 20
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
- name: approximate
  type: string
  doc: 'Gelu approximation algorithm: `"tanh"`, `"none"`(default).`"none"`: do not
    use approximation.`"tanh"`: use tanh approximation.'
  required: false
  default: none
---

Gelu takes one input data (Tensor<T>) and produces one
output data (Tensor<T>) where the gaussian error linear units function,
$y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
If the attribute "approximate" is set to "tanh", the function estimation,
$y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
to the tensor elementwise.
