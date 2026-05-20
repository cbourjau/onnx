---
inherit: v7/sub
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to high-precision numeric tensors.
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
    doc: First operand.
    differentiable: true
  - name: B
    type: T
    doc: Second operand.
    differentiable: true
outputs:
  update:
  - name: C
    type: T
    doc: Result, has same element type as two inputs
    differentiable: true
---
