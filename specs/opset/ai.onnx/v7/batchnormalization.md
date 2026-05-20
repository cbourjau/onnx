---
inherit: v6/batchnormalization
since_version: 7
inputs:
  update:
  - name: scale
    type: T
    doc: If spatial is true, the dimension of scale is (C). If spatial is false, the
      dimensions of scale are (C x D1 x ... x Dn)
  - name: B
    type: T
    doc: If spatial is true, the dimension of bias is (C). If spatial is false, the
      dimensions of bias are (C x D1 x ... x Dn)
  - name: mean
    type: T
    doc: If spatial is true, the dimension of the running mean (training) or the estimated
      mean (testing) is (C). If spatial is false, the dimensions of the running mean
      (training) or the estimated mean (testing) are (C x D1 x ... x Dn).
  - name: var
    type: T
    doc: If spatial is true, the dimension of the running variance(training) or the
      estimated variance (testing) is (C). If spatial is false, the dimensions of
      the running variance(training) or the estimated variance (testing) are (C x
      D1 x ... x Dn).
outputs:
  update:
  - name: Y
    type: T
    doc: The output tensor of the same shape as X
  - name: mean
    type: T
    doc: The running mean after the BatchNormalization operator.
    kind: Option
  - name: var
    type: T
    doc: The running variance after the BatchNormalization operator.
    kind: Option
  - name: saved_mean
    type: T
    doc: Saved mean used during training to speed up gradient computation.
    kind: Option
  - name: saved_var
    type: T
    doc: Saved variance used during training to speed up gradient computation.
    kind: Option
attributes:
  remove:
  - is_test
  update:
  - name: epsilon
    type: float
    doc: The epsilon value to use to avoid division by zero.
    required: false
    default: 9.999999747378752e-06
  - name: momentum
    type: float
    doc: Factor used in computing the running mean and variance.e.g., running_mean
      = running_mean * momentum + mean * (1 - momentum).
    required: false
    default: 0.8999999761581421
  - name: spatial
    type: int
    doc: If true, compute the mean and variance across per activation. If false, compute
      the mean and variance across per feature over each mini-batch.
    required: false
    default: 1
---

Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
    there are multiple cases for the number of outputs, which we list below:

    Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
    Output case #2: Y (test mode)
        This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
