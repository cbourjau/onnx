---
op: OneHotEncoder
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
  - tensor(string)
inputs:
- name: X
  type: T
  doc: Data to be encoded.
outputs:
- name: Y
  type: tensor(float)
  doc: Encoded output data, having one more dimension than X.
attributes:
- name: cats_int64s
  type: ints
  doc: List of categories, ints.<br>One and only one of the 'cats_*' attributes must
    be defined.
  required: false
- name: cats_strings
  type: strings
  doc: List of categories, strings.<br>One and only one of the 'cats_*' attributes
    must be defined.
  required: false
- name: zeros
  type: int
  doc: If true and category is not present, will return all zeros; if false and a
    category if not found, the operator will fail.
  required: false
  default: 1
---

Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8,
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.
