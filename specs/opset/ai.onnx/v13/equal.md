---
inherit: v11/equal
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input types to all numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(bool)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
inputs:
  update:
  - name: A
    type: T
    doc: First input operand for the logical operator.
    differentiable: false
  - name: B
    type: T
    doc: Second input operand for the logical operator.
    differentiable: false
outputs:
  update:
  - name: C
    type: T1
    doc: Result tensor.
    differentiable: false
---
