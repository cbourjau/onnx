---
inherit: v11/concat
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain output types to any tensor type.
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
  - name: inputs
    type: T
    doc: List of tensors for concatenation
    kind:
      homogeneous: true
      min_arity: 1
    differentiable: true
outputs:
  update:
  - name: concat_result
    type: T
    doc: Concatenated tensor
    differentiable: true
---
