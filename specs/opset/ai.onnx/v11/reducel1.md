---
inherit: v1/reducel1
since_version: 11
attributes:
  update:
  - name: axes
    type: ints
    doc: A list of integers, along which to reduce. The default is to reduce over
      all the dimensions of the input tensor. Accepted range is [-r, r-1] where r
      = rank(data).
    required: false
---

Computes the L1 norm of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
