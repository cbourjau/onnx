---
op: ConstantOfShape
domain: ai.onnx
since_version: 9
type_constraints:
- name: T1
  doc: Constrain input types.
  allowed:
  - tensor(int64)
- name: T2
  doc: Constrain output types to be numerics.
  allowed:
  - tensor(bool)
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
- name: input
  type: T1
  doc: 1D tensor. The shape of the expected output tensor. If empty tensor is given,
    the output would be a scalar. All values must be >= 0.
outputs:
- name: output
  type: T2
  doc: Output tensor of shape specified by 'input'.If attribute 'value' is specified,
    the value and datatype of the output tensor is taken from 'value'.If attribute
    'value' is not specified, the value in the output defaults to 0, and the datatype
    defaults to float32.
attributes:
- name: value
  type: tensor
  doc: (Optional) The value of the output elements.Should be a one-element tensor.
    If not specified, it defaults to a tensor of value 0 and datatype float32
  required: false
---

Generate a tensor with given value and shape.
