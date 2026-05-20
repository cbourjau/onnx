---
op: TopK
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: I
  doc: Constrain index tensor to int64
  allowed:
  - tensor(int64)
inputs:
- name: X
  type: T
  doc: Tensor of shape [a_0, a_1, ..., a_{n-1}]
outputs:
- name: Values
  type: T
  doc: Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing
    top K values from the input tensor
- name: Indices
  type: I
  doc: Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing
    the corresponding input tensor indices for the top K values.
attributes:
- name: k
  type: int
  doc: Number of top elements to retrieve
  required: true
- name: axis
  type: int
  doc: Dimension on which to do the sort.
  required: false
  default: -1
---

Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:
  -Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
   contains the indices of the top k elements (original indices from the input
   tensor).
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
