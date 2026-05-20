---
inherit: v6/sum
since_version: 8
inputs:
  update:
  - name: data_0
    type: T
    doc: List of tensors for sum.
    kind:
      homogeneous: true
      min_arity: 1
outputs:
  update:
  - name: sum
    type: T
    doc: Output tensor.
---

Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
