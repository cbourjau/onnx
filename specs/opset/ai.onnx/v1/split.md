---
op: Split
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: input
  type: T
  doc: The tensor to split
- name: split
  type: T
  doc: Optional list of output lengths (see also arg 'split')
  kind: Option
outputs:
- name: outputs...
  type: T
  doc: One or more outputs forming list of tensors after splitting
  kind:
    homogeneous: true
    min_arity: 1
attributes:
- name: axis
  type: int
  doc: Which axis to split on
  required: false
- name: split
  type: ints
  doc: length of each output
  required: false
---

Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
