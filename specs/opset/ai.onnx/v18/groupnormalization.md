---
op: GroupNormalization
domain: ai.onnx
since_version: 18
deprecated: true
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(bfloat16)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: X
  type: T
  doc: Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where
    `N` is the batch size, `C` is the number of channels, and `H` and `W` are the
    height and width of the data. Statistics are computed for every group of channels
    over `C`, `H`, and `W`. For non-image cases, the dimensions are in the form of
    `(N x C x D1 x D2 ... Dn)`.
  differentiable: true
- name: scale
  type: T
  doc: Scale tensor of shape `(num_groups)`.
  differentiable: true
- name: bias
  type: T
  doc: Bias tensor of shape `(num_groups)`.
  differentiable: true
outputs:
- name: Y
  type: T
  doc: The output tensor of the same shape as `X`.
  differentiable: true
attributes:
- name: num_groups
  type: int
  doc: The number of groups of channels. It should be a divisor of the number of channels
    `C`.
  required: true
- name: epsilon
  type: float
  doc: The epsilon value to use to avoid division by zero.
  required: false
  default: 9.999999747378752e-06
---

A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
