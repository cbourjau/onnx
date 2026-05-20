---
inherit: v11/flatten
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output to all tensor types.
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
  - name: input
    type: T
    doc: A tensor of rank >= axis.
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: A 2D tensor with the contents of the input tensor, with input dimensions
      up to axis flattened to the outer dimension of the output and remaining input
      dimensions flattened into the inner dimension of the output.
    differentiable: true
---
