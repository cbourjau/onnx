---
inherit: v11/softmax
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

      Describes the dimension Softmax will be performed on.

      Negative value means counting dimensions

      from the back. Accepted range is [-r, r-1] where r = rank(input).

      '
    required: false
    default: -1
---

The operator computes the normalized exponential values for the given input:

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

The "axis" attribute indicates the dimension along which Softmax
will be performed. The output tensor has the same shape
and contains the Softmax values of the corresponding input.
