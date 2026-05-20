---
inherit: v6/prelu
since_version: 7
inputs:
  update:
  - name: slope
    type: T
    doc: Slope tensor. The shape of slope can be smaller than first input X; if so,
      its shape must be unidirectional broadcastable to X
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor (same size as X)
---

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
