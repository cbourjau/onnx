---
op: Concat
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: inputs
  type: T
  doc: List of tensors for concatenation
  kind:
    homogeneous: true
    min_arity: 1
outputs:
- name: concat_result
  type: T
  doc: Concatenated tensor
attributes:
- name: axis
  type: int
  doc: Which axis to concat on.  Default value is 1.
  required: false
---

Concatenate a list of tensors into a single tensor
