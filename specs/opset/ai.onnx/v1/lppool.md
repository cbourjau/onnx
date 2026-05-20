---
op: LpPool
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
  doc: Input data tensor from the previous operator; dimensions for image case are
    (N x C x H x W), where N is the batch size, C is the number of channels, and H
    and W are the height and the width of the data. For non image case, the dimension
    are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
outputs:
- name: Y
  type: T
  doc: Output data tensor from Lp pooling across the input tensor. Dimensions will
    vary based on various kernel, stride, and pad sizes.
attributes:
- name: auto_pad
  type: string
  doc: 'auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default
    value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER
    mean pad the input so that the output size match the input.In case of odd number
    add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
    VALID mean no padding. DEPRECATION NOTE: auto_pad is only intended to support
    legacy uses, and for framework authors, one is explicitly encouraged to use explicit
    padding specified in the pads attribute.'
  required: false
  default: NOTSET
- name: kernel_shape
  type: ints
  doc: The size of the kernel along each axis.
  required: false
- name: p
  type: float
  doc: p value of the Lp norm used to pool over the input data, default is 2.0.
  required: false
  default: 2.0
- name: pads
  type: ints
  doc: Padding for the beginning and ending along each axis, it can take any value
    greater than or equal to 0. The value represent the number of pixels added to
    the beginning and end part of the corresponding axis. `pads` format should be
    as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number
    of pixels added at the beginning of axis `i` and xi_end, the number of pixels
    added at the end of axis `i`. This attribute cannot be used simultaneously with
    auto_pad attribute.
  required: false
- name: strides
  type: ints
  doc: Stride along each axis.
  required: false
---

LpPool consumes an input tensor X and applies Lp pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.
