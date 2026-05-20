---
op: RandomNormalLike
domain: ai.onnx
since_version: 1
deterministic: false
type_constraints:
- name: T1
  doc: Constrain to any tensor type. If the dtype attribute is not provided this must
    be a valid output type.
  allowed:
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
- name: T2
  doc: Constrain output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
inputs:
- name: input
  type: T1
  doc: Input tensor to copy shape and optionally type information from.
outputs:
- name: output
  type: T2
  doc: Output tensor of random values drawn from normal distribution
attributes:
- name: dtype
  type: int
  doc: (Optional) The data type for the elements of the output tensor, if not specified,
    we will use the data type of the input tensor.
  required: false
- name: mean
  type: float
  doc: The mean of the normal distribution.
  required: false
  default: 0.0
- name: scale
  type: float
  doc: The standard deviation of the normal distribution.
  required: false
  default: 1.0
- name: seed
  type: float
  doc: (Optional) Seed to the random generator, if not specified we will auto generate
    one.
  required: false
---

Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
