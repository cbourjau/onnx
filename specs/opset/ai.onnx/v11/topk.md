---
inherit: v10/topk
since_version: 11
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to numeric tensors.
    allowed:
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
  - name: X
    type: T
    doc: Tensor of shape [a_0, a_1, ..., a_{n-1}]
    differentiable: true
  - name: K
    type: tensor(int64)
    doc: A 1-D tensor containing a single positive value corresponding to the number
      of top elements to retrieve
    differentiable: false
outputs:
  update:
  - name: Values
    type: T
    doc: Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing
      top K values from the input tensor
    differentiable: true
  - name: Indices
    type: I
    doc: Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing
      the corresponding input tensor indices for the top K values.
    differentiable: false
attributes:
  add:
  - name: largest
    type: int
    doc: Whether to return the top-K largest or smallest elements.
    required: false
    default: 1
  - name: sorted
    type: int
    doc: Whether to return the elements in sorted order.
    required: false
    default: 1
  update:
  - name: axis
    type: int
    doc: Dimension on which to do the sort. Negative value means counting dimensions
      from the back. Accepted range is [-r, r-1] where r = rank(input).
    required: false
    default: -1
---

Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:

* Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
  which contains the values of the top k elements along the specified axis
* Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
  contains the indices of the top k elements (original indices from the input
  tensor).

* If "largest" is 1 (the default value) then the k largest elements are returned.
* If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
* If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
a tiebreaker. That is, the element with the lower index will appear first.
