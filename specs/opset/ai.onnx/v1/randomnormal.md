---
op: RandomNormal
domain: ai.onnx
since_version: 1
deterministic: false
type_constraints:
- name: T
  doc: Constrain output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
outputs:
- name: output
  type: T
  doc: Output tensor of random values drawn from normal distribution
attributes:
- name: shape
  type: ints
  doc: The shape of the output tensor.
  required: true
- name: dtype
  type: int
  doc: The data type for the elements of the output tensor. Default is TensorProto::FLOAT.
  required: false
  default: 1
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

Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
