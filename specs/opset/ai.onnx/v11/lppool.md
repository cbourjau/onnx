---
inherit: v2/lppool
since_version: 11
inputs:
  update:
  - name: X
    type: T
    doc: Input data tensor from the previous operator; dimensions for image case are
      (N x C x H x W), where N is the batch size, C is the number of channels, and
      H and W are the height and the width of the data. For non image case, the dimensions
      are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output data tensor from Lp pooling across the input tensor. Dimensions will
      vary based on various kernel, stride, and pad sizes.
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
