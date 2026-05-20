---
inherit: v1/greater
since_version: 7
inputs:
  update:
  - name: A
    type: T
    doc: First input operand for the logical operator.
  - name: B
    type: T
    doc: Second input operand for the logical operator.
attributes:
  remove:
  - axis
  - broadcast
---

Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
