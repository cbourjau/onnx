---
inherit: v11/argmin
since_version: 12
attributes:
  add:
  - name: select_last_index
    type: int
    doc: Whether to select the last index or the first index if the {name} appears
      in multiple indices, default is False (first index).
    required: false
    default: 0
---

Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
