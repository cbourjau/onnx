---
inherit: v12/max
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to numeric tensors.
    allowed:
    - tensor(bfloat16)
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
  - name: data_0
    type: T
    doc: List of tensors for max.
    kind:
      homogeneous: true
      min_arity: 1
    differentiable: true
outputs:
  update:
  - name: max
    type: T
    doc: Output tensor.
    differentiable: true
---
