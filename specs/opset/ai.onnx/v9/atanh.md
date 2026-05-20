---
op: Atanh
domain: ai.onnx
since_version: 9
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
  doc: The hyperbolic arctangent values of the input tensor computed element-wise
  differentiable: true
---

Calculates the hyperbolic arctangent of the given input tensor element-wise.
