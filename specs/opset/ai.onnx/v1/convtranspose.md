---
op: ConvTranspose
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
  doc: Input data tensor from previous layer; has size (N x C x H x W), where N is
    the batch size, C is the number of channels, and H and W are the height and width.
    Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ...
    x Dn)
- name: W
  type: T
  doc: The weight tensor that will be used in the convolutions; has size (C x M/group
    x kH x kW), where C is the number of channels, and kH and kW are the height and
    width of the kernel, and M is the number of feature maps. For more than 2 dimensions,
    the weight shape will be (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x
    ... x kn) is the dimension of the kernel. The number of channels in the output
    should be equal to W.shape[1] * group (assuming zero based indices of the shape
    array)
- name: B
  type: T
  doc: Optional 1D bias to be added to the convolution, has size of M.
  kind: Option
outputs:
- name: Y
  type: T
  doc: Output data tensor that contains the result of the convolution. The output
    dimensions are functions of the kernel size, stride size, pad lengths and group
    count. The number of channels in the output should be equal to W.shape[1] * group
    (assuming zero based indices of the shape array)
attributes:
- name: auto_pad
  type: string
  doc: auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default
    value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER
    mean pad the input so that the output spatial size match the input.In case of
    odd number add the extra padding at the end for SAME_UPPER and at the beginning
    for SAME_LOWER. VALID mean no padding.
  required: false
  default: NOTSET
- name: dilations
  type: ints
  doc: dilation value along each spatial axis of the filter.
  required: false
- name: group
  type: int
  doc: number of groups input channels and output channels are divided into.
  required: false
  default: 1
- name: kernel_shape
  type: ints
  doc: The shape of the convolution kernel. If not present, should be inferred from
    input W.
  required: false
- name: output_padding
  type: ints
  doc: The zero-padding added to one side of the output. This is also called adjs/adjustment
    in some frameworks.
  required: false
- name: output_shape
  type: ints
  doc: The shape of the output can be explicitly set which will cause pads values
    to be auto generated. If output_shape is specified pads values are ignored. See
    doc for details for equations to generate pads
  required: false
- name: pads
  type: ints
  doc: Padding for the beginning and ending along each spatial axis, it can take any
    value greater than or equal to 0. The value represent the number of pixels added
    to the beginning and end part of the corresponding axis. `pads` format should
    be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number
    of pixels added at the beginning of axis `i` and xi_end, the number of pixels
    added at the end of axis `i`. This attribute cannot be used simultaneously with
    auto_pad attribute. If not present, the padding defaults to 0 along start and
    end of each spatial axis.
  required: false
- name: strides
  type: ints
  doc: Stride along each spatial axis.
  required: false
---

The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
