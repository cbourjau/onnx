---
inherit: v10/nonmaxsuppression
since_version: 11
inputs:
  update:
  - name: iou_threshold
    type: tensor(float)
    doc: Float representing the threshold for deciding whether boxes overlap too much
      with respect to IOU. Boxes with IoU strictly greater than this threshold are
      suppressed. It is scalar. Value range [0, 1]. Default to 0.
    kind: Option
---
