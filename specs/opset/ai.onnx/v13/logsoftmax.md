---
inherit: v11/logsoftmax
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

      Describes the dimension LogSoftmax will be performed on.

      Negative value means counting dimensions

      from the back. Accepted range is [-r, r-1] where r = rank(input).

      '
    required: false
    default: -1
---

The operator computes the log of softmax values for the given input:

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The "axis" attribute indicates the dimension along which LogSoftmax
will be performed. The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.
