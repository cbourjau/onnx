---
inherit: v1/or
since_version: 7
inputs:
  update:
  - name: A
    type: T
    doc: First input operand for the logical operator.
    differentiable: false
  - name: B
    type: T
    doc: Second input operand for the logical operator.
    differentiable: false
outputs:
  update:
  - name: C
    type: T1
    doc: Result tensor.
    differentiable: false
attributes:
  remove:
  - axis
  - broadcast
---

Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
