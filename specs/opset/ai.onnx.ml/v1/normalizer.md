---
op: Normalizer
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
  doc: Data to be encoded, a tensor of shape [N,C] or [C]
outputs:
- name: Y
  type: tensor(float)
  doc: Encoded output data
attributes:
- name: norm
  type: string
  doc: One of 'MAX,' 'L1,' 'L2'
  required: false
  default: MAX
---

Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.
