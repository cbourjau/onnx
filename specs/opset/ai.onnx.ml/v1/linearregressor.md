---
op: LinearRegressor
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
  doc: Data to be regressed.
outputs:
- name: Y
  type: tensor(float)
  doc: Regression outputs (one per target, per example).
attributes:
- name: coefficients
  type: floats
  doc: Weights of the model(s).
  required: false
- name: intercepts
  type: floats
  doc: Weights of the intercepts, if used.
  required: false
- name: post_transform
  type: string
  doc: Indicates the transform to apply to the regression output vector.<br>One of
    'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
  required: false
  default: NONE
- name: targets
  type: int
  doc: The total number of regression targets, 1 if not defined.
  required: false
  default: 1
---

Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.
