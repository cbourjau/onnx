---
op: Binarizer
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T
  doc: The input must be a tensor of a numeric type. The output will be of the same
    tensor type.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: X
  type: T
  doc: Data to be binarized
outputs:
- name: Y
  type: T
  doc: Binarized output data
attributes:
- name: threshold
  type: float
  doc: Values greater than this are mapped to 1, others to 0.
  required: false
  default: 0.0
---

Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
