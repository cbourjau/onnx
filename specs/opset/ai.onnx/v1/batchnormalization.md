---
op: BatchNormalization
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: X
  type: T
  doc: The input 4-dimensional tensor of shape NCHW.
- name: scale
  type: T
  doc: The scale as a 1-dimensional tensor of size C to be applied to the output.
- name: B
  type: T
  doc: The bias as a 1-dimensional tensor of size C to be applied to the output.
- name: mean
  type: T
  doc: The running mean (training) or the estimated mean (testing) as a 1-dimensional
    tensor of size C.
- name: var
  type: T
  doc: The running variance (training) or the estimated variance (testing) as a 1-dimensional
    tensor of size C.
outputs:
- name: Y
  type: T
  doc: The output 4-dimensional tensor of the same shape as X.
- name: mean
  type: T
  doc: The running mean after the BatchNormalization operator. Must be in-place with
    the input mean. Should not be used for testing.
  kind: Option
- name: var
  type: T
  doc: The running variance after the BatchNormalization operator. Must be in-place
    with the input var. Should not be used for testing.
  kind: Option
- name: saved_mean
  type: T
  doc: Saved mean used during training to speed up gradient computation. Should not
    be used for testing.
  kind: Option
- name: saved_var
  type: T
  doc: Saved variance used during training to speed up gradient computation. Should
    not be used for testing.
  kind: Option
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: true
- name: epsilon
  type: float
  doc: The epsilon value to use to avoid division by zero, default is 1e-5f.
  required: false
  default: 9.999999747378752e-06
- name: is_test
  type: int
  doc: If set to nonzero, run spatial batch normalization in test mode, default is
    0.
  required: false
  default: 0
- name: momentum
  type: float
  doc: Factor used in computing the running mean and variance.e.g., running_mean =
    running_mean * momentum + mean * (1 - momentum), default is 0.9f.
  required: false
  default: 0.8999999761581421
- name: spatial
  type: int
  doc: If true, compute the mean and variance across all spatial elements If false,
    compute the mean and variance across per feature.Default is 1.
  required: false
  default: 1
---

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
