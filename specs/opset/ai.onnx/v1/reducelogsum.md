---
op: ReduceLogSum
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to high-precision numeric tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int32)
  - tensor(int64)
  - tensor(uint32)
  - tensor(uint64)
inputs:
- name: data
  type: T
  doc: An input tensor.
outputs:
- name: reduced
  type: T
  doc: Reduced output tensor.
attributes:
- name: axes
  type: ints
  doc: A list of integers, along which to reduce. The default is to reduce over all
    the dimensions of the input tensor.
  required: false
- name: keepdims
  type: int
  doc: Keep the reduced dimension or not, default 1 means keep reduced dimension.
  required: false
  default: 1
---

Computes the log sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
