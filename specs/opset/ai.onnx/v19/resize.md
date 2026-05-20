---
inherit: v18/resize
since_version: 19
attributes:
  update:
  - name: coordinate_transformation_mode
    type: string
    doc: "\nThis attribute describes how to transform the coordinate in the resized\
      \ tensor to the coordinate in the original tensor.\n\nThe coordinate of each\
      \ dimension is transformed individually. Let's describe a case using axis x\
      \ as an example.\nDenote `x_resized` as the coordinate of axis x in the resized\
      \ tensor,\n `x_original` as the coordinate of axis x in the original tensor,\n\
      \ `length_original` as the length of the original tensor in axis x,\n `length_resized`\
      \ as the length of the resized tensor in axis x,\n `scale = length_resized /\
      \ length_original`,\n `output_width` the target length on the axis x which can\
      \ be a fractional number when it is calculated out of a scale factor,\n and\
      \ `output_width_int` the effective output width as an integer.\n\nif coordinate_transformation_mode\
      \ is `\"half_pixel\"`,\n```\nx_original = (x_resized + 0.5) / scale - 0.5\n\
      ```\n\nif coordinate_transformation_mode is `\"half_pixel_symmetric\"`,\n```\n\
      adjustment = output_width_int / output_width\ncenter = input_width / 2\noffset\
      \ = center * (1 - adjustment)\nx_ori = offset + (x + 0.5) / scale - 0.5\n```\n\
      \nif coordinate_transformation_mode is `\"pytorch_half_pixel\"`,\n```\nx_original\
      \ = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0\n```\n\nif coordinate_transformation_mode\
      \ is `\"align_corners\"`,\n```\nx_original = x_resized * (length_original -\
      \ 1) / (length_resized - 1)\n```\n\nif coordinate_transformation_mode is `\"\
      asymmetric\"`,\n```\nx_original = x_resized / scale\n```\n\nif coordinate_transformation_mode\
      \ is `\"tf_crop_and_resize\"`,\n```\nx_original = length_resized > 1 ? start_x\
      \ * (length_original - 1) + x_resized * (end_x - start_x) * (length_original\
      \ - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original -\
      \ 1)\n```\n."
    required: false
    default: half_pixel
  - name: keep_aspect_ratio_policy
    type: string
    doc: '

      This attribute describes how to interpret the `sizes` input with regard to keeping
      the original aspect ratio of the input, and it is not applicable when

      the `scales` input is used.


      Given a set of `sizes`, associated with a subset of `axes` (explicitly provided
      or default), and assuming `d = axes[i]`, with `i` being the index of the provided
      `sizes`.


      If `keep_aspect_ratio_policy` is `"stretch"`, the original aspect ratio is disregarded,
      and the input is resized to the specified size:

      `out_size[d] = sizes[i]`


      If `keep_aspect_ratio_policy` is `"not_larger"`, the sizes are adjusted so that
      no extent of the output is larger than the specified size, while keeping the
      original aspect ratio:

      ```

      scale = Min(sizes[i] / in_size[d])

      out_size[d] = round_int(scale * in_size[d])

      ```


      If `keep_aspect_ratio_policy` is `"not_smaller"`, the sizes are adjusted so
      that no extent of the output is smaller than the specified size, while keeping
      the original aspect ratio:

      ```

      scale = Max(sizes[i] / in_size[d])

      out_size[d] = round_int(scale * in_size[d])

      ```


      For non-resizable axes (those not specified in `axes`), the output size will
      be equal to the input size.


      Note: `round_int` stands for computing the nearest integer value, rounding halfway
      cases up.'
    required: false
    default: stretch
---

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
```
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
```
if input \"sizes\" is not specified.
