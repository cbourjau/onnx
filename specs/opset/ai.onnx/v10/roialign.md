---
op: RoiAlign
domain: ai.onnx
since_version: 10
type_constraints:
- name: T1
  doc: Constrain types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T2
  doc: Constrain types to int tensors.
  allowed:
  - tensor(int64)
inputs:
- name: X
  type: T1
  doc: Input data tensor from the previous operator; 4-D feature map of shape (N,
    C, H, W), where N is the batch size, C is the number of channels, and H and W
    are the height and the width of the data.
- name: rois
  type: T1
  doc: RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois,
    4) given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate
    system of the input image. Each coordinate set has a 1:1 correspondence with the
    'batch_indices' input.
- name: batch_indices
  type: T2
  doc: 1-D tensor of shape (num_rois,) with each element denoting the index of the
    corresponding image in the batch.
outputs:
- name: Y
  type: T1
  doc: RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width).
    The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th
    RoI X[r-1].
attributes:
- name: mode
  type: string
  doc: 'The pooling method. Two modes are supported: ''avg'' and ''max''. Default
    is ''avg''.'
  required: false
  default: avg
- name: output_height
  type: int
  doc: default 1; Pooled output Y's height.
  required: false
  default: 1
- name: output_width
  type: int
  doc: default 1; Pooled output Y's width.
  required: false
  default: 1
- name: sampling_ratio
  type: int
  doc: Number of sampling points in the interpolation grid used to compute the output
    value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio
    grid points are used. If == 0, then an adaptive number of grid points are used
    (computed as ceil(roi_width / output_width), and likewise for height). Default
    is 0.
  required: false
  default: 0
- name: spatial_scale
  type: float
  doc: 'Multiplicative spatial scale factor to translate ROI coordinates from their
    input spatial scale to the scale used when pooling, i.e., spatial scale of the
    input feature map X relative to the input image. E.g.; default is 1.0f. '
  required: false
  default: 1.0
---

Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
