---
op: LpNormalization
domain: ai.onnx
since_version: 1
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
  doc: Input matrix
  differentiable: true
outputs:
- name: output
  type: T
  doc: Matrix after normalization
  differentiable: true
attributes:
- name: axis
  type: int
  doc: The axis on which to apply normalization, -1 mean last axis.
  required: false
  default: -1
- name: p
  type: int
  doc: The order of the normalization, only 1 or 2 are supported.
  required: false
  default: 2
---

Given a matrix, apply Lp-normalization along the provided axis.
The output is computed as: `output = input / Lp_norm(input, axis)`.
When the Lp norm is zero (i.e., all elements along the axis are zero),
the output is defined to be zero to avoid division by zero.
