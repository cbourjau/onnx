---
inherit: v1/split
since_version: 2
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all tensor types.
    allowed:
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
  remove:
  - split
outputs:
  add:
  - name: outputs
    type: T
    doc: One or more outputs forming list of tensors after splitting
    kind:
      homogeneous: true
      min_arity: 1
  remove:
  - outputs...
attributes:
  update:
  - name: axis
    type: int
    doc: 'Which axis to split on. '
    required: false
    default: 0
---

Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
