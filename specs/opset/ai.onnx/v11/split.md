---
inherit: v2/split
since_version: 11
attributes:
  update:
  - name: axis
    type: int
    doc: Which axis to split on. A negative value means counting dimensions from the
      back. Accepted range is [-rank, rank-1] where r = rank(input).
    required: false
    default: 0
  - name: split
    type: ints
    doc: length of each output. Values should be >= 0.
    required: false
---
