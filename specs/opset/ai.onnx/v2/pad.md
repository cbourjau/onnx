---
inherit: v1/pad
since_version: 2
attributes:
  add:
  - name: pads
    type: ints
    doc: List of integers indicating the number of padding elements to add or remove
      (if negative) at the beginning and end of each axis. For 2D it is the number
      of pixels. `pads` rank should be double of the input's rank. `pads` format should
      be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number
      of pixels added at the beginning of axis `i` and xi_end, the number of pixels
      added at the end of axis `i`.
    required: true
  remove:
  - paddings
  update:
  - name: value
    type: float
    doc: One float, indicates the value to be filled.
    required: false
    default: 0.0
---

Given `data` tensor, pads, mode, and value.
Example:
  Insert 0 pads to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]
