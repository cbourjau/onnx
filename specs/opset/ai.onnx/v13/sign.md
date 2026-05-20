---
inherit: v9/sign
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all numeric tensors.
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
  - name: input
    type: T
    doc: Input tensor
    differentiable: false
outputs:
  update:
  - name: output
    type: T
    doc: The sign of the input tensor computed element-wise. It has the same shape
      and type of the input.
    differentiable: false
---
