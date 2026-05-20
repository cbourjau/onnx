---
inherit: v7/prelu
since_version: 9
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float/int tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(uint32)
    - tensor(uint64)
inputs:
  update:
  - name: X
    type: T
    doc: Input tensor
    differentiable: true
  - name: slope
    type: T
    doc: Slope tensor. The shape of slope can be smaller than first input X; if so,
      its shape must be unidirectional broadcastable to X
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor (same size as X)
    differentiable: true
---
