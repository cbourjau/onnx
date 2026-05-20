---
inherit: v1/instancenormalization
since_version: 6
inputs:
  update:
  - name: input
    type: T
    doc: Input data tensor from the previous operator; dimensions for image case are
      (N x C x H x W), where N is the batch size, C is the number of channels, and
      H and W are the height and the width of the data. For non image case, the dimensions
      are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
    differentiable: true
  - name: scale
    type: T
    doc: The input 1-dimensional scale tensor of size C.
    differentiable: true
  - name: B
    type: T
    doc: The input 1-dimensional bias tensor of size C.
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: The output tensor of the same shape as input.
    differentiable: true
attributes:
  remove:
  - consumed_inputs
  update:
  - name: epsilon
    type: float
    doc: The epsilon value to use to avoid division by zero.
    required: false
    default: 9.999999747378752e-06
---
