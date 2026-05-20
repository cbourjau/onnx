---
op: Scaler
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T
  doc: The input must be a tensor of a numeric type.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: X
  type: T
  doc: Data to be scaled.
outputs:
- name: Y
  type: tensor(float)
  doc: Scaled output data.
attributes:
- name: offset
  type: floats
  doc: First, offset by this.<br>Can be length of features in an [N,F] tensor or length
    1, in which case it applies to all features, regardless of dimension count.
  required: false
- name: scale
  type: floats
  doc: Second, multiply by this.<br>Can be length of features in an [N,F] tensor or
    length 1, in which case it applies to all features, regardless of dimension count.<br>Must
    be same length as 'offset'
  required: false
---

Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
