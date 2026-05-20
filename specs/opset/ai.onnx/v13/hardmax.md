---
inherit: v11/hardmax
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
inputs:
  update:
  - name: input
    type: T
    doc: The input tensor of rank >= axis.
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: The output values with the same shape as the input tensor.
    differentiable: true
attributes:
  update:
  - name: axis
    type: int
    doc: '

      Describes the dimension Hardmax will be performed on.

      Negative value means counting dimensions

      from the back. Accepted range is [-r, r-1] where r = rank(input).

      '
    required: false
    default: -1
---

The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The "axis" attribute indicates the dimension along which Hardmax
will be performed. The output tensor has the same shape
and contains the Hardmax values of the corresponding input.
