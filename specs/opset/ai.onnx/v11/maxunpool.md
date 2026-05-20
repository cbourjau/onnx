---
inherit: v9/maxunpool
since_version: 11
inputs:
  update:
  - name: X
    type: T1
    doc: Input data tensor that has to be unpooled. This tensor is typically the first
      output of the MaxPool op.Dimensions for image case are (N x C x H x W), where
      N is the batch size, C is the number of channels, and H and W are the height
      and the width of the data. For non-image case, the dimensions are in the form
      of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension
      denotation is in effect, the operation expects the input data tensor to arrive
      with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE
      ...].
    differentiable: true
  - name: I
    type: T2
    doc: Input data tensor containing the indices corresponding to elements in the
      first input tensor X.This tensor is typically the second output of the MaxPool
      op.Dimensions must be the same as input tensor X. The indices are linear, i.e.
      computed considering the tensor as flattened 1-D tensor, assuming row-major
      storage. Also, the linear indices should not consider padding. So the values
      in indices are in the range [0, N x C x D1 x ... x Dn).
    differentiable: false
  - name: output_shape
    type: T2
    doc: The shape of the output can be explicitly set which will cause pads values
      to be auto generated. If 'output_shape' is specified, 'pads' values are ignored.
    kind: Option
    differentiable: false
outputs:
  update:
  - name: output
    type: T1
    doc: Output data tensor that contains the result of the unpooling.
    differentiable: true
attributes:
  update:
  - name: strides
    type: ints
    doc: Stride along each spatial axis. If not present, the stride defaults to 1
      along each spatial axis.
    required: false
---
