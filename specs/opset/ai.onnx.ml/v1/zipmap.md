---
op: ZipMap
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T
  doc: The output will be a sequence of string or integer maps to float.
  allowed:
  - seq(map(int64, float))
  - seq(map(string, float))
inputs:
- name: X
  type: tensor(float)
  doc: The input values
outputs:
- name: Z
  type: T
  doc: The output map
attributes:
- name: classlabels_int64s
  type: ints
  doc: The keys when using int keys.<br>One and only one of the 'classlabels_*' attributes
    must be defined.
  required: false
- name: classlabels_strings
  type: strings
  doc: The keys when using string keys.<br>One and only one of the 'classlabels_*'
    attributes must be defined.
  required: false
---

Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
