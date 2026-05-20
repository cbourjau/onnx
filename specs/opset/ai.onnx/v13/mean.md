---
inherit: v8/mean
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
inputs:
  update:
  - name: data_0
    type: T
    doc: List of tensors for mean.
    kind:
      homogeneous: true
      min_arity: 1
    differentiable: true
outputs:
  update:
  - name: mean
    type: T
    doc: Output tensor.
    differentiable: true
---
