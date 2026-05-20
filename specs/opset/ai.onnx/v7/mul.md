---
inherit: v6/mul
since_version: 7
inputs:
  update:
  - name: A
    type: T
    doc: First operand.
  - name: B
    type: T
    doc: Second operand.
outputs:
  update:
  - name: C
    type: T
    doc: Result, has same element type as two inputs
attributes:
  remove:
  - axis
  - broadcast
---

Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
