---
inherit: v1/pow
since_version: 7
inputs:
  update:
  - name: X
    type: T
    doc: First operand, base of the exponent.
  - name: Y
    type: T
    doc: Second operand, power of the exponent.
outputs:
  update:
  - name: Z
    type: T
    doc: Output tensor.
attributes:
  remove:
  - axis
  - broadcast
---

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
