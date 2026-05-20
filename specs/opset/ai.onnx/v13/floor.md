---
inherit: v6/floor
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
  - name: X
    type: T
    doc: Input tensor
    differentiable: false
outputs:
  update:
  - name: Y
    type: T
    doc: Output tensor
    differentiable: false
---

Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
