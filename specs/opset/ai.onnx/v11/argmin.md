---
inherit: v1/argmin
since_version: 11
attributes:
  update:
  - name: axis
    type: int
    doc: The axis in which to compute the arg indices. Accepted range is [-r, r-1]
      where r = rank(data).
    required: false
    default: 0
---

Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
The input tensor must not be empty.
The type of the output tensor is integer.
