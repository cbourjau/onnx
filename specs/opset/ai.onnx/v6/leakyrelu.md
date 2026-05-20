---
inherit: v1/leakyrelu
since_version: 6
inputs:
  update:
  - name: X
    type: T
    doc: Input tensor
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor
    differentiable: true
attributes:
  remove:
  - consumed_inputs
  update:
  - name: alpha
    type: float
    doc: Coefficient of leakage.
    required: false
    default: 0.009999999776482582
---
