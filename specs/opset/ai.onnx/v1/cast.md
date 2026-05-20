---
op: Cast
domain: ai.onnx
since_version: 1
type_constraints:
- name: T1
  doc: Constrain input types. Casting from strings and complex are not supported.
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
- name: T2
  doc: Constrain output types. Casting to strings and complex are not supported.
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
  doc: Input tensor to be cast.
outputs:
- name: output
  type: T2
  doc: Output tensor with the same shape as input with type specified by the 'to'
    argument
attributes:
- name: to
  type: string
  doc: The data type to which the elements of the input tensor are cast. Strictly
    must be one of the types from DataType enum in TensorProto
  required: true
---

The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
