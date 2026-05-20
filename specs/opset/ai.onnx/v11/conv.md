---
inherit: v1/conv
since_version: 11
inputs:
  update:
  - name: X
    type: T
    doc: Input data tensor from previous layer; has size (N x C x H x W), where N
      is the batch size, C is the number of channels, and H and W are the height and
      width. Note that this is for the 2D image. Otherwise the size is (N x C x D1
      x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation
      expects input data tensor to arrive with the dimension denotation of [DATA_BATCH,
      DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
    differentiable: true
  - name: W
    type: T
    doc: The weight tensor that will be used in the convolutions; has size (M x C/group
      x kH x kW), where C is the number of channels, and kH and kW are the height
      and width of the kernel, and M is the number of feature maps. For more than
      2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where
      (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension
      denotation is in effect, the operation expects the weight tensor to arrive with
      the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL,
      FILTER_SPATIAL ...]. Assuming zero based indices for the shape array, X.shape[1]
      == (W.shape[1] * group) == C and W.shape[0] mod G == 0. Or in other words FILTER_IN_CHANNEL
      multiplied by the number of groups should be equal to DATA_CHANNEL and the number
      of feature maps M should be a multiple of the number of groups G.
    differentiable: true
  - name: B
    type: T
    doc: Optional 1D bias to be added to the convolution, has size of M.
    kind: Option
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output data tensor that contains the result of the convolution. The output
      dimensions are functions of the kernel size, stride size, and pad lengths.
    differentiable: true
attributes:
  update:
  - name: auto_pad
    type: string
    doc: auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default
      value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER
      mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])`
      for each axis `i`. The padding is split between the two sides equally or almost
      equally (depending on whether it is even or odd). In case the padding is an
      odd number, the extra padding is added at the end for SAME_UPPER and at the
      beginning for SAME_LOWER.
    required: false
    default: NOTSET
  - name: dilations
    type: ints
    doc: dilation value along each spatial axis of the filter. If not present, the
      dilation defaults is 1 along each spatial axis.
    required: false
  - name: strides
    type: ints
    doc: Stride along each spatial axis. If not present, the stride defaults is 1
      along each spatial axis.
    required: false
---
