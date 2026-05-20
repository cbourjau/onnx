---
op: InstanceNormalization
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
- name: input
  type: T
  doc: The input 4-dimensional tensor of shape NCHW.
- name: scale
  type: T
  doc: The input 1-dimensional scale tensor of size C.
- name: B
  type: T
  doc: The input 1-dimensional bias tensor of size C.
outputs:
- name: output
  type: T
  doc: The output 4-dimensional tensor of the same shape as input.
attributes:
- name: consumed_inputs
  type: ints
  doc: legacy optimization attribute.
  required: false
- name: epsilon
  type: float
  doc: The epsilon value to use to avoid division by zero, default is 1e-5f.
  required: false
  default: 9.999999747378752e-06
---

Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.
