---
op: ConvInteger
domain: ai.onnx
since_version: 10
type_constraints:
- name: T1
  doc: Constrain input x and its zero point data type to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
- name: T2
  doc: Constrain input w and its zero point data type to 8-bit integer tensor.
  allowed:
  - tensor(int8)
  - tensor(uint8)
- name: T3
  doc: Constrain output y data type to 32-bit integer tensor.
  allowed:
  - tensor(int32)
inputs:
- name: x
  type: T1
  doc: Input data tensor from previous layer; has size (N x C x H x W), where N is
    the batch size, C is the number of channels, and H and W are the height and width.
    Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ...
    x Dn). Optionally, if dimension denotation is in effect, the operation expects
    input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL,
    DATA_FEATURE, DATA_FEATURE ...].
- name: w
  type: T2
  doc: 'The weight tensor that will be used in the convolutions; has size (M x C/group
    x kH x kW), where C is the number of channels, and kH and kW are the height and
    width of the kernel, and M is the number of feature maps. For more than 2 dimensions,
    the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x
    ... kn) is the dimension of the kernel. Optionally, if dimension denotation is
    in effect, the operation expects the weight tensor to arrive with the dimension
    denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL
    ...]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for
    the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. '
- name: x_zero_point
  type: T1
  doc: Zero point tensor for input 'x'. It's optional and default value is 0. It's
    a scalar, which means a per-tensor/layer quantization.
  kind: Option
- name: w_zero_point
  type: T2
  doc: Zero point tensor for input 'w'. It's optional and default value is 0.  It
    could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output
    channel quantization. If it's a 1-D tensor, its number of elements should be equal
    to the number of output channels (M)
  kind: Option
outputs:
- name: y
  type: T3
  doc: Output data tensor that contains the result of the convolution. The output
    dimensions are functions of the kernel size, stride size, and pad lengths.
attributes:
- name: auto_pad
  type: string
  doc: auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default
    value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER
    mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])`
    for each axis `i`. The padding is split between the two sides equally or almost
    equally (depending on whether it is even or odd). In case the padding is an odd
    number, the extra padding is added at the end for SAME_UPPER and at the beginning
    for SAME_LOWER.
  required: false
  default: NOTSET
- name: dilations
  type: ints
  doc: dilation value along each spatial axis of the filter. If not present, the dilation
    defaults to 1 along each axis.
  required: false
- name: group
  type: int
  doc: number of groups input channels and output channels are divided into. default
    is 1.
  required: false
  default: 1
- name: kernel_shape
  type: ints
  doc: The shape of the convolution kernel. If not present, should be inferred from
    input 'w'.
  required: false
- name: pads
  type: ints
  doc: Padding for the beginning and ending along each spatial axis, it can take any
    value greater than or equal to 0.The value represent the number of pixels added
    to the beginning and end part of the corresponding axis.`pads` format should be
    as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number
    ofpixels added at the beginning of axis `i` and xi_end, the number of pixels added
    at the end of axis `i`.This attribute cannot be used simultaneously with auto_pad
    attribute. If not present, the padding defaultsto 0 along start and end of each
    spatial axis.
  required: false
- name: strides
  type: ints
  doc: Stride along each spatial axis. If not present, the stride defaults to 1 along
    each axis.
  required: false
---

The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
