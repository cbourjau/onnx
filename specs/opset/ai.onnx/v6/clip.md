---
inherit: v1/clip
since_version: 6
attributes:
  remove:
  - consumed_inputs
  update:
  - name: max
    type: float
    doc: Maximum value, above which element is replaced by max
    required: false
    default: 3.4028234663852886e+38
  - name: min
    type: float
    doc: Minimum value, under which element is replaced by min
    required: false
    default: -3.4028234663852886e+38
---
