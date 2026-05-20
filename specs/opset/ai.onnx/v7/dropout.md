---
inherit: v6/dropout
since_version: 7
outputs:
  update:
  - name: mask
    type: T
    doc: The output mask.
    kind: Option
attributes:
  remove:
  - is_test
  update:
  - name: ratio
    type: float
    doc: The ratio of random dropout
    required: false
    default: 0.5
---

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
