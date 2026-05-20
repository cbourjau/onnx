---
inherit: v10/maxpool
since_version: 11
attributes:
  update:
  - name: dilations
    type: ints
    doc: Dilation value along each spatial axis of filter. If not present, the dilation
      defaults to 1 along each spatial axis.
    required: false
  - name: strides
    type: ints
    doc: Stride along each spatial axis. If not present, the stride defaults to 1
      along each spatial axis.
    required: false
---
