---
inherit: v16/gridsample
since_version: 20
inputs:
  update:
  - name: X
    type: T1
    doc: Input tensor of rank r+2 that has shape (N, C, D1, D2, ..., Dr), where N
      is the batch size, C is the number of channels, D1, D2, ..., Dr are the spatial
      dimensions.
    differentiable: true
  - name: grid
    type: T2
    doc: Input offset of shape (N, D1_out, D2_out, ..., Dr_out, r), where D1_out,
      D2_out, ..., Dr_out are the spatial dimensions of the grid and output, and r
      is the number of spatial dimensions. Grid specifies the sampling locations normalized
      by the input spatial dimensions. Therefore, it should have most values in the
      range of [-1, 1]. If the grid has values outside the range of [-1, 1], the corresponding
      outputs will be handled as defined by padding_mode. Following computer vision
      convention, the coordinates in the length-r location vector are listed from
      the innermost tensor dimension to the outermost, the opposite of regular tensor
      indexing.
    differentiable: false
outputs:
  update:
  - name: Y
    type: T1
    doc: Output tensor of rank r+2 that has shape (N, C, D1_out, D2_out, ..., Dr_out)
      of the sampled values. For integer input types, intermediate values are computed
      as floating point and cast to integer at the end.
    differentiable: true
attributes:
  update:
  - name: align_corners
    type: int
    doc: If align_corners=1, the extrema (-1 and 1) are considered as referring to
      the center points of the input's corner pixels (voxels, etc.). If align_corners=0,
      they are instead considered as referring to the corner points of the input's
      corner pixels (voxels, etc.), making the sampling more resolution agnostic.
    required: false
    default: 0
  - name: mode
    type: string
    doc: 'Three interpolation modes: linear (default), nearest and cubic. The "linear"
      mode includes linear and N-linear interpolation modes depending on the number
      of spatial dimensions of the input tensor (i.e. linear for 1 spatial dimension,
      bilinear for 2 spatial dimensions, etc.). The "cubic" mode also includes N-cubic
      interpolation modes following the same rules. The "nearest" mode rounds to the
      nearest even index when the sampling point falls halfway between two indices.'
    required: false
    default: linear
---

Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
the `grid` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, ..., Dr_out).

The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, ..., dr_in).
The (n, d1_out, d2_out, ..., dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values
at the (n, c, d1_out, d2_out, ..., dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode)
and a padding mode (for `grid` positions falling outside the 2-dimensional image).

For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[n, c, h_out, w_out]`.

The GridSample operator is often used in doing grid generator and sampler in the
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
