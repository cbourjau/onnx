---
inherit: v11/split
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all tensor types.
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
  add:
  - name: split
    type: tensor(int64)
    doc: Optional length of each output. Values should be >= 0.Sum of the values must
      be equal to the dim value at 'axis' specified.
    kind: Option
    differentiable: false
  update:
  - name: input
    type: T
    doc: The tensor to split
    differentiable: true
outputs:
  update:
  - name: outputs
    type: T
    doc: One or more outputs forming list of tensors after splitting
    kind:
      homogeneous: true
      min_arity: 1
    differentiable: true
attributes:
  remove:
  - split
---

Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.
