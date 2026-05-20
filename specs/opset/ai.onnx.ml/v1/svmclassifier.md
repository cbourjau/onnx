---
op: SVMClassifier
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input must be a tensor of a numeric type, either [C] or [N,C].
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
- name: T2
  doc: The output type will be a tensor of strings or integers, depending on which
    of the classlabels_* attributes is used. Its size will match the batch size of
    the input.
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
  doc: Class scores (one per class per example), if prob_a and prob_b are provided
    they are probabilities for each class, otherwise they are raw scores.
attributes:
- name: classlabels_ints
  type: ints
  doc: Class labels if using integer labels.<br>One and only one of the 'classlabels_*'
    attributes must be defined.
  required: false
- name: classlabels_strings
  type: strings
  doc: Class labels if using string labels.<br>One and only one of the 'classlabels_*'
    attributes must be defined.
  required: false
- name: coefficients
  type: floats
  doc: ''
  required: false
- name: kernel_params
  type: floats
  doc: List of 3 elements containing gamma, coef0, and degree, in that order. Zero
    if unused for the kernel.
  required: false
- name: kernel_type
  type: string
  doc: The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.
  required: false
  default: LINEAR
- name: post_transform
  type: string
  doc: Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,'
    'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
  required: false
  default: NONE
- name: prob_a
  type: floats
  doc: First set of probability coefficients.
  required: false
- name: prob_b
  type: floats
  doc: Second set of probability coefficients. This array must be same size as prob_a.<br>If
    these are provided then output Z are probability estimates, otherwise they are
    raw scores.
  required: false
- name: rho
  type: floats
  doc: ''
  required: false
- name: support_vectors
  type: floats
  doc: ''
  required: false
- name: vectors_per_class
  type: ints
  doc: ''
  required: false
---

Support Vector Machine classifier
