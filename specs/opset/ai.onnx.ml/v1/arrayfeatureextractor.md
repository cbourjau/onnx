---
op: ArrayFeatureExtractor
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T
  doc: The input must be a tensor of a numeric type or string. The output will be
    of the same tensor type.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
  - tensor(string)
inputs:
- name: X
  type: T
  doc: Data to be selected
- name: Y
  type: tensor(int64)
  doc: The indices, based on 0 as the first index of any dimension.
outputs:
- name: Z
  type: T
  doc: Selected output data as an array
---

Select elements of the input tensor based on the indices passed.<br>
    The indices are applied to the last axes of the tensor.
