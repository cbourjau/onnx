---
inherit: v1/topk
since_version: 10
inputs:
  add:
  - name: K
    type: tensor(int64)
    doc: A 1-D tensor containing a single positive value corresponding to the number
      of top elements to retrieve
attributes:
  remove:
  - k
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
