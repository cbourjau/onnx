---
op: Flatten
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: input
  type: T
  doc: A tensor of rank >= axis.
outputs:
- name: output
  type: T
  doc: A 2D tensor with the contents of the input tensor, with input dimensions up
    to axis flattened to the outer dimension of the output and remaining input dimensions
    flattened into the inner dimension of the output.
attributes:
- name: axis
  type: int
  doc: 'Indicate up to which input dimensions (exclusive) should be flattened to the
    outer dimension of the output. The value for axis must be in the range [0, R],
    where R is the rank of the input tensor. When axis = 0, the shape of the output
    tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0,
    d_1, ... d_n). '
  required: false
  default: 1
---

Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
