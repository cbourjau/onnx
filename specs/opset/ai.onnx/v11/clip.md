---
inherit: v6/clip
since_version: 11
inputs:
  add:
  - name: min
    type: T
    doc: Minimum value, under which element is replaced by min. It must be a scalar(tensor
      of empty shape).
    kind: Option
  - name: max
    type: T
    doc: Maximum value, above which element is replaced by max. It must be a scalar(tensor
      of empty shape).
    kind: Option
attributes:
  remove:
  - max
  - min
---

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
