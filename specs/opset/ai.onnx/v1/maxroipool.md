---
op: MaxRoiPool
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
    and W are the height and the width of the data.
  differentiable: true
- name: rois
  type: T
  doc: RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois,
    5) given as [[batch_id, x1, y1, x2, y2], ...].
  differentiable: false
outputs:
- name: Y
  type: T
  doc: RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0],
    pooled_shape[1]).
  differentiable: true
attributes:
- name: pooled_shape
  type: ints
  doc: ROI pool output shape (height, width).
  required: true
- name: spatial_scale
  type: float
  doc: Multiplicative spatial scale factor to translate ROI coordinates from their
    input scale to the scale used when pooling.
  required: false
  default: 1.0
---

ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).
