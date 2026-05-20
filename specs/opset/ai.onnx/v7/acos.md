---
op: Acos
domain: ai.onnx
since_version: 7
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
  doc: The arccosine of the input tensor computed element-wise
  differentiable: true
---

Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
