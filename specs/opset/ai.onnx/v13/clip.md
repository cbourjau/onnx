---
inherit: v12/clip
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
inputs:
  update:
  - name: input
    type: T
    doc: Input tensor whose elements to be clipped
    differentiable: true
  - name: min
    type: T
    doc: Minimum value, under which element is replaced by min. It must be a scalar(tensor
      of empty shape).
    kind: Option
    differentiable: false
  - name: max
    type: T
    doc: Maximum value, above which element is replaced by max. It must be a scalar(tensor
      of empty shape).
    kind: Option
    differentiable: false
outputs:
  update:
  - name: output
    type: T
    doc: Output tensor with clipped input elements
    differentiable: true
---

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
When 'min' is greater than 'max', the clip operator sets all the 'input' values to
the value of 'max'. Thus, this is equivalent to 'Min(max, Max(input, min))'.
