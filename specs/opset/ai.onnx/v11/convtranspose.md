---
inherit: v1/convtranspose
since_version: 11
inputs:
  update:
  - name: X
    type: T
    doc: Input data tensor from previous layer; has size (N x C x H x W), where N
      is the batch size, C is the number of channels, and H and W are the height and
      width. Note that this is for the 2D image. Otherwise the size is (N x C x D1
      x D2 ... x Dn)
    differentiable: true
  - name: W
    type: T
    doc: The weight tensor that will be used in the convolutions; has size (C x M/group
      x kH x kW), where C is the number of channels, and kH and kW are the height
      and width of the kernel, and M is the number of feature maps. For more than
      2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn), where
      (k1 x k2 x ... x kn) is the dimension of the kernel. The number of channels
      in the output should be equal to W.shape[1] * group (assuming zero based indices
      of the shape array)
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
      dimensions are functions of the kernel size, stride size, pad lengths and group
      count. The number of channels in the output should be equal to W.shape[1] *
      group (assuming zero based indices of the shape array)
    differentiable: true
attributes:
  update:
  - name: auto_pad
    type: string
    doc: auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default
      value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER
      mean pad the input so that `output_shape[i] = input_shape[i] * strides[i]` for
      each axis `i`. The padding is split between the two sides equally or almost
      equally (depending on whether it is even or odd). In case the padding is an
      odd number, the extra padding is added at the end for SAME_UPPER and at the
      beginning for SAME_LOWER.
    required: false
    default: NOTSET
  - name: dilations
    type: ints
    doc: dilation value along each spatial axis of the filter. If not present, the
      dilation defaults to 1 along each spatial axis.
    required: false
  - name: output_padding
    type: ints
    doc: Additional elements added to the side with higher coordinate indices in the
      output. Each padding value in "output_padding" must be less than the corresponding
      stride/dilation dimension. By default, this attribute is a zero vector. Note
      that this attribute doesn't directly affect the computed output values. It only
      controls the selection of the computed values, so changing this attribute only
      adds or removes output elements. If "output_shape" is explicitly provided, "output_padding"
      does not contribute additional size to "output_shape" but participates in the
      computation of the needed padding amount. This is also called adjs or adjustment
      in some frameworks.
    required: false
  - name: output_shape
    type: ints
    doc: The shape of the output can be explicitly set which will cause pads values
      to be auto generated. If output_shape is specified pads values are ignored.
      See doc for details for equations to generate pads. Note that the output_shape
      attribute value should not include dimensions for batch size and channels, which
      are automatically inferred.
    required: false
  - name: strides
    type: ints
    doc: Stride along each spatial axis. If not present, the stride defaults to 1
      along each spatial axis.
    required: false
---

The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
