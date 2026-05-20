---
inherit: v11/resize
since_version: 13
type_constraints:
  update:
  - name: T1
    doc: Constrain input 'X' and output 'Y' to all tensor types.
    allowed:
    - tensor(bfloat16)
    - tensor(bool)
    - tensor(complex128)
    - tensor(complex64)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(string)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
inputs:
  update:
  - name: X
    type: T1
    doc: N-D tensor
    differentiable: true
  - name: roi
    type: T2
    doc: 1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the
      rank of X. The RoIs' coordinates are normalized in the coordinate system of
      the input image. It only takes effect when coordinate_transformation_mode is
      "tf_crop_and_resize"
    kind: Option
    differentiable: false
  - name: scales
    type: tensor(float)
    doc: The scale array along each dimension. It takes value greater than 0. If it's
      less than 1, it's sampling down, otherwise, it's upsampling. The number of elements
      of 'scales' should be the same as the rank of input 'X'. One of 'scales' and
      'sizes' MUST be specified and it is an error if both are specified. If 'sizes'
      is needed, the user can use an empty string as the name of 'scales' in this
      operator's input list.
    kind: Option
    differentiable: false
  - name: sizes
    type: tensor(int64)
    doc: The size of the output tensor. The number of elements of 'sizes' should be
      the same as the rank of input 'X'. Only one of 'scales' and 'sizes' can be specified.
    kind: Option
    differentiable: false
outputs:
  update:
  - name: Y
    type: T1
    doc: N-D tensor after resizing
    differentiable: true
attributes:
  update:
  - name: coordinate_transformation_mode
    type: string
    doc: '

      This attribute describes how to transform the coordinate in the resized tensor
      to the coordinate in the original tensor. <br/>


      The coordinate of each dimension is transformed individually. Let''s describe
      a case using axis x as an example.

      Denote x_resized as the coordinate of axis x in the resized tensor, x_original
      as the coordinate of axis x in the original tensor, length_original as the length
      of the original tensor in axis x, length_resized as the length of the resized
      tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi", scale
      = length_resized / length_original, <br/>


      if coordinate_transformation_mode is "half_pixel", <br/>

      x_original = (x_resized + 0.5) / scale - 0.5, <br/>


      if coordinate_transformation_mode is "pytorch_half_pixel", <br/>

      x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0, <br/>


      if coordinate_transformation_mode is "align_corners", <br/>

      x_original = x_resized * (length_original - 1) / (length_resized - 1), <br/>


      if coordinate_transformation_mode is "asymmetric", <br/>

      x_original = x_resized / scale, <br/>


      if coordinate_transformation_mode is "tf_crop_and_resize", <br/>

      x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized
      * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x
      + end_x) * (length_original - 1).'
    required: false
    default: half_pixel
---
