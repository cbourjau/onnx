---
inherit: v7/dropout
since_version: 10
type_constraints:
  add:
  - name: T1
    doc: Constrain output mask types to boolean tensors.
    allowed:
    - tensor(bool)
outputs:
  update:
  - name: mask
    type: T1
    doc: The output mask.
    kind: Option
---

Dropout takes one input floating tensor and produces two tensor outputs,
output (floating tensor) and mask (`Tensor<bool>`). Depending on whether it is
in test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
