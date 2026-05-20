---
inherit: v11/scatternd
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to any tensor type.
    allowed:
    - tensor(bfloat16)
    - tensor(bool)
    - tensor(complex128)
    - tensor(complex64)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(string)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
inputs:
  update:
  - name: data
    type: T
    doc: Tensor of rank r >= 1.
    differentiable: true
  - name: indices
    type: tensor(int64)
    doc: Tensor of rank q >= 1.
    differentiable: false
  - name: updates
    type: T
    doc: Tensor of rank q + r - indices_shape[-1] - 1.
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: Tensor of rank r >= 1.
    differentiable: true
---
