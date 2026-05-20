---
op: Upsample
domain: ai.onnx
since_version: 1
support_level: experimental
type_constraints:
- name: T
  doc: Constrain output types to bool, int32, int64, float16, float, double tensors.
  allowed:
  - tensor(bool)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: X
  type: T
  doc: 4-D tensor, [N,C,H,W]
outputs:
- name: Y
  type: T
  doc: 4-D tensor after resizing, [N,C,H,W]
attributes:
- name: height_scale
  type: float
  doc: The scale along height dimension. It takes value greater than or equal to 1.
  required: true
- name: width_scale
  type: float
  doc: The scale along width dimension. It takes value greater than or equal to 1.
  required: true
- name: mode
  type: string
  doc: 'Two interpolation modes: nearest(default), bilinear'
  required: false
  default: nearest
---

Upsample the input tensor.
The width and height of the output tensor are:
  output_width = floor(input_width * width_scale),
  output_height = floor(input_height * height_scale).
Example:
  Given `data` tensor, width_scale, height_scale, mode,
  Upsample the input 4-D tensor in nearest mode:
  data = [[[
      [1, 2],
      [3, 4]
  ]]]
  width_scale = 2
  height_scale = 2
  mode = "nearest"
  output = [[[
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4]
  ]]]
