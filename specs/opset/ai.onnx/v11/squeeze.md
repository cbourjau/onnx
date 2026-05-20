---
inherit: v1/squeeze
since_version: 11
attributes:
  update:
  - name: axes
    type: ints
    doc: List of integers indicating the dimensions to squeeze. Negative value means
      counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
    required: false
---
