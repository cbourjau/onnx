---
inherit: v6/mean
since_version: 8
inputs:
  update:
  - name: data_0
    type: T
    doc: List of tensors for mean.
    kind:
      homogeneous: true
      min_arity: 1
outputs:
  update:
  - name: mean
    type: T
    doc: Output tensor.
---

Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
