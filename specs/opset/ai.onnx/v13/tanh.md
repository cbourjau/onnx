---
inherit: v6/tanh
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
  - name: input
    type: T
    doc: Input tensor
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: The hyperbolic tangent values of the input tensor computed element-wise
    differentiable: true
---
