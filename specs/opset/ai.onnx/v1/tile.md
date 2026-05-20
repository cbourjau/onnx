---
op: Tile
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T1
  doc: Constrain tiles and axis's type to int64 tensors.
  allowed:
  - tensor(int64)
inputs:
- name: input
  type: T
  doc: Input tensor of any shape.
- name: tiles
  type: T
  doc: Number of repeated copies to make of the input tensor.
- name: axis
  type: T
  doc: Axis along which to repeat.
outputs:
- name: output
  type: T
  doc: Output tensor of same shape and type as input.
---

Repeat the elements of a tensor along an axis.
