---
inherit: v1/reducemin
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
