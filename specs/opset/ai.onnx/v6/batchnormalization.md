---
inherit: v1/batchnormalization
since_version: 6
inputs:
  update:
  - name: X
    type: T
    doc: Input data tensor from the previous operator; dimensions for image case are
      (N x C x H x W), where N is the batch size, C is the number of channels, and
      H and W are the height and the width of the data. For non image case, the dimensions
      are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
outputs:
  update:
  - name: Y
    type: T
    doc: The output tensor of the same shape as X.
attributes:
  remove:
  - consumed_inputs
---
