---
op: LinearClassifier
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input must be a tensor of a numeric type, and of shape [N,C] or [C]. In
    the latter case, it will be treated as [1,C]
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
- name: T2
  doc: The output will be a tensor of strings or integers.
  allowed:
  - tensor(int64)
  - tensor(string)
inputs:
- name: X
  type: T1
  doc: Data to be classified.
outputs:
- name: Y
  type: T2
  doc: Classification outputs (one class per example).
- name: Z
  type: tensor(float)
  doc: Classification scores ([N,E] - one score for each class and example
attributes:
- name: coefficients
  type: floats
  doc: A collection of weights of the model(s).
  required: true
- name: classlabels_ints
  type: ints
  doc: Class labels when using integer labels. One and only one 'classlabels' attribute
    must be defined.
  required: false
- name: classlabels_strings
  type: strings
  doc: Class labels when using string labels. One and only one 'classlabels' attribute
    must be defined.
  required: false
- name: intercepts
  type: floats
  doc: A collection of intercepts.
  required: false
- name: multi_class
  type: int
  doc: Indicates whether to do OvR or multinomial (0=OvR is the default).
  required: false
  default: 0
- name: post_transform
  type: string
  doc: Indicates the transform to apply to the scores vector.<br>One of 'NONE,' 'SOFTMAX,'
    'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
  required: false
  default: NONE
---

Linear classifier
