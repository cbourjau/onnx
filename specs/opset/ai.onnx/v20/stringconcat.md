---
op: StringConcat
domain: ai.onnx
since_version: 20
type_constraints:
- name: T
  doc: Inputs and outputs must be UTF-8 strings
  allowed:
  - tensor(string)
inputs:
- name: X
  type: T
  doc: Tensor to prepend in concatenation
  differentiable: false
- name: Y
  type: T
  doc: Tensor to append in concatenation
  differentiable: false
outputs:
- name: Z
  type: T
  doc: Concatenated string tensor
  differentiable: false
---

StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support)
