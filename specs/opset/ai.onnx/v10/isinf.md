---
op: IsInf
domain: ai.onnx
since_version: 10
type_constraints:
- name: T1
  doc: Constrain input types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
- name: T2
  doc: Constrain output types to boolean tensors.
  allowed:
  - tensor(bool)
inputs:
- name: X
  type: T1
  doc: input
  differentiable: false
outputs:
- name: Y
  type: T2
  doc: output
  differentiable: false
attributes:
- name: detect_negative
  type: int
  doc: (Optional) Whether map negative infinity to true. Default to 1 so that negative
    infinity induces true. Set this attribute to 0 if negative infinity should be
    mapped to false.
  required: false
  default: 1
- name: detect_positive
  type: int
  doc: (Optional) Whether map positive infinity to true. Default to 1 so that positive
    infinity induces true. Set this attribute to 0 if positive infinity should be
    mapped to false.
  required: false
  default: 1
---

Map infinity to true and other values to false.
