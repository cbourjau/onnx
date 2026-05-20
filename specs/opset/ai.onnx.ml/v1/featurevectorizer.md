---
op: FeatureVectorizer
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input type must be a tensor of a numeric type.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: X
  type: T1
  doc: An ordered collection of tensors, all with the same element type.
  kind:
    homogeneous: true
    min_arity: 1
outputs:
- name: Y
  type: tensor(float)
  doc: The output array, elements ordered as the inputs.
attributes:
- name: inputdimensions
  type: ints
  doc: The size of each input in the input list
  required: false
---

Concatenates input tensors into one continuous output.<br>
    All input shapes are 2-D and are concatenated along the second dimension. 1-D tensors are treated as [1,C].
    Inputs are copied to the output maintaining the order of the input arguments.<br>
    All inputs must be integers or floats, while the output will be all floating point values.
