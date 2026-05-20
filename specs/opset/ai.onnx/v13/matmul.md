---
inherit: v9/matmul
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float/int tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int32)
    - tensor(int64)
    - tensor(uint32)
    - tensor(uint64)
inputs:
  update:
  - name: A
    type: T
    doc: N-dimensional matrix A
    differentiable: true
  - name: B
    type: T
    doc: N-dimensional matrix B
    differentiable: true
outputs:
  update:
  - name: Y
    type: T
    doc: Matrix multiply results from A * B
    differentiable: true
---
