---
op: SVMRegressor
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T
  doc: The input type must be a tensor of a numeric type, either [C] or [N,C].
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: X
  type: T
  doc: Data to be regressed.
outputs:
- name: Y
  type: tensor(float)
  doc: Regression outputs (one score per target per example).
attributes:
- name: coefficients
  type: floats
  doc: Support vector coefficients.
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
- name: n_supports
  type: int
  doc: The number of support vectors.
  required: false
  default: 0
- name: one_class
  type: int
  doc: Flag indicating whether the regression is a one-class SVM or not.
  required: false
  default: 0
- name: post_transform
  type: string
  doc: Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,'
    'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'
  required: false
  default: NONE
- name: rho
  type: floats
  doc: ''
  required: false
- name: support_vectors
  type: floats
  doc: Chosen support vectors
  required: false
---

Support Vector Machine regression prediction and one-class SVM anomaly detection.
