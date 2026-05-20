---
op: IsNaN
domain: ai.onnx
since_version: 9
type_constraints:
- name: T1
  doc: Constrain input types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T2
  doc: Constrain output types to boolean tensors.
  allowed:
  - tensor(bool)
inputs:
- name: X
  type: T1
  doc: input
outputs:
- name: Y
  type: T2
  doc: output
---

Returns which elements of the input are NaN.
