---
inherit: v13/resize
since_version: 18
inputs:
  update:
  - name: roi
    type: T2
    doc: 1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the
      rank of X or the length of axes, if provided. The RoIs' coordinates are normalized
      in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode
      is "tf_crop_and_resize"
    kind: Option
    differentiable: false
  - name: scales
    type: tensor(float)
    doc: The scale array along each dimension. It takes value greater than 0. If it's
      less than 1, it's sampling down, otherwise, it's upsampling. The number of elements
      of 'scales' should be the same as the rank of input 'X' or the length of 'axes',
      if provided. One of 'scales' and 'sizes' MUST be specified and it is an error
      if both are specified. If 'sizes' is needed, the user can use an empty string
      as the name of 'scales' in this operator's input list.
    kind: Option
    differentiable: false
  - name: sizes
    type: tensor(int64)
    doc: 'Target size of the output tensor. Its interpretation depends on the ''keep_aspect_ratio_policy''
      value.The number of elements of ''sizes'' should be the same as the rank of
      input ''X'', or the length of ''axes'', if provided. Only one of ''scales''
      and ''sizes'' can be specified. '
    kind: Option
    differentiable: false
attributes:
  add:
  - name: antialias
    type: int
    doc: If set to 1, "linear" and "cubic" interpolation modes will use an antialiasing
      filter when downscaling. Antialiasing is achieved by stretching the resampling
      filter by a factor max(1, 1 / scale), which means that when downsampling, more
      input pixels contribute to an output pixel.
    required: false
    default: 0
  - name: axes
    type: ints
    doc: If provided, it specifies a subset of axes that 'roi', 'scales' and 'sizes'
      refer to. If not provided, all axes are assumed [0, 1, ..., r-1], where r =
      rank(data). Non-specified dimensions are interpreted as non-resizable. Negative
      value means counting dimensions from the back. Accepted range is [-r, r-1],
      where r = rank(data). Behavior is undefined if an axis is repeated.
    required: false
  - name: keep_aspect_ratio_policy
    type: string
    doc: '

      This attribute describes how to interpret the `sizes` input with regard to keeping
      the original aspect ratio of the input, and it is not applicable when

      the `scales` input is used. <br/>


      Given a set of `sizes`, associated with a subset of `axes` (explicitly provided
      or default), and assuming `d = axes[i]`, with `i` being the index of the provided
      `sizes`. <br/>


      If `keep_aspect_ratio_policy` is `"stretch"`, the original aspect ratio is disregarded,
      and the input is resized to the specified size: <br/>

      `out_size[d] = sizes[i]` <br/>


      If `keep_aspect_ratio_policy` is `"not_larger"`, the sizes are adjusted so that
      no extent of the output is larger than the specified size, while keeping the
      original aspect ratio: <br/>

      `scale = Min(sizes[i] / in_size[d])` <br/>

      `out_size[d] = round_int(scale * in_size[d])` <br/>


      If `keep_aspect_ratio_policy` is `"not_smaller"`, the sizes are adjusted so
      that no extent of the output is smaller than the specified size, while keeping
      the original aspect ratio: <br/>

      `scale = Max(sizes[i] / in_size[d])` <br/>

      `out_size[d] = round_int(scale * in_size[d])` <br/>


      For non-resizable axes (those not specified in `axes`), the output size will
      be equal to the input size.


      Note: `round_int` stands for computing the nearest integer value, rounding halfway
      cases up.'
    required: false
    default: stretch
  update:
  - name: coordinate_transformation_mode
    type: string
    doc: '

      This attribute describes how to transform the coordinate in the resized tensor
      to the coordinate in the original tensor. <br/>


      The coordinate of each dimension is transformed individually. Let''s describe
      a case using axis x as an example.

      Denote x_resized as the coordinate of axis x in the resized tensor, x_original
      as the coordinate of axis x in the original tensor, `length_original` as the
      length of the original tensor in axis x, length_resized as the length of the
      resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi",
      `scale = length_resized / length_original`, <br/>


      if coordinate_transformation_mode is `"half_pixel"`, <br/>

      `x_original = (x_resized + 0.5) / scale - 0.5` <br/>


      if coordinate_transformation_mode is `"pytorch_half_pixel"`, <br/>

      `x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0` <br/>


      if coordinate_transformation_mode is `"align_corners"`, <br/>

      `x_original = x_resized * (length_original - 1) / (length_resized - 1)` <br/>


      if coordinate_transformation_mode is `"asymmetric"`, <br/>

      `x_original = x_resized / scale` <br/>


      if coordinate_transformation_mode is `"tf_crop_and_resize"`, <br/>

      `x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized
      * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x
      + end_x) * (length_original - 1)`

      .'
    required: false
    default: half_pixel
  - name: cubic_coeff_a
    type: float
    doc: The coefficient 'a' used in cubic interpolation. Two common choice are -0.5
      (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out Equation (4)
      in https://ieeexplore.ieee.org/document/1163711 for the details. This attribute
      is valid only if mode is "cubic".
    required: false
    default: -0.75
  - name: mode
    type: string
    doc: 'Three interpolation modes: "nearest" (default), "linear" and "cubic". The
      "linear" mode includes linear interpolation for 1D tensor and N-linear interpolation
      for N-D tensor (for example, bilinear interpolation for 2D tensor). The "cubic"
      mode includes cubic interpolation for 1D tensor and N-cubic interpolation for
      N-D tensor (for example, bicubic interpolation for 2D tensor).'
    required: false
    default: nearest
  - name: nearest_mode
    type: string
    doc: 'Four modes: "round_prefer_floor" (default, as known as round half down),
      "round_prefer_ceil" (as known as round half up), "floor", "ceil". Only used
      by nearest interpolation. It indicates how to get "nearest" pixel in input tensor
      from x_original, so this attribute is valid only if "mode" is "nearest".'
    required: false
    default: round_prefer_floor
---

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is: <br/>
  `output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)` <br/>
if input \"sizes\" is not specified.
