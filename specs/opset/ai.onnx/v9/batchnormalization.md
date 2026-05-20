---
inherit: v7/batchnormalization
since_version: 9
inputs:
  update:
  - name: X
    type: T
    doc: Input data tensor from the previous operator; dimensions are in the form
      of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of channels.
      Statistics are computed for every channel of C over N and D1 to Dn dimensions.
      For image data, input dimensions become (N x C x H x W). The op also accepts
      single dimension input of size N in which case C is assumed to be 1
    differentiable: true
  - name: scale
    type: T
    doc: Scale tensor of shape (C).
    differentiable: true
  - name: B
    type: T
    doc: Bias tensor of shape (C).
    differentiable: true
  - name: mean
    type: T
    doc: running (training) or estimated (testing) mean tensor of shape (C).
    differentiable: true
  - name: var
    type: T
    doc: running (training) or estimated (testing) variance tensor of shape (C).
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: The output tensor of the same shape as X
    differentiable: true
  - name: mean
    type: T
    doc: The running mean after the BatchNormalization operator.
    kind: Option
    differentiable: false
  - name: var
    type: T
    doc: The running variance after the BatchNormalization operator.
    kind: Option
    differentiable: false
  - name: saved_mean
    type: T
    doc: Saved mean used during training to speed up gradient computation.
    kind: Option
    differentiable: false
  - name: saved_var
    type: T
    doc: Saved variance used during training to speed up gradient computation.
    kind: Option
    differentiable: false
attributes:
  remove:
  - spatial
---

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
