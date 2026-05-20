---
inherit: v10/averagepool
since_version: 11
inputs:
  update:
  - name: X
    type: T
    doc: Input data tensor from the previous operator; dimensions for image case are
      (N x C x H x W), where N is the batch size, C is the number of channels, and
      H and W are the height and the width of the data. For non image case, the dimensions
      are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally,
      if dimension denotation is in effect, the operation expects the input data tensor
      to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE,
      DATA_FEATURE ...].
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output data tensor from average or max pooling across the input tensor. Dimensions
      will vary based on various kernel, stride, and pad sizes. Floor value of the
      dimension is used
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
  - name: strides
    type: ints
    doc: Stride along each spatial axis. If not present, the stride defaults to 1
      along each spatial axis.
    required: false
---

AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
or when ceil_mode is disabled:
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])
 ```

 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
