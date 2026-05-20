---
inherit: v9/scan
since_version: 11
attributes:
  update:
  - name: scan_input_axes
    type: ints
    doc: An optional list of M flags. The i-th element of the list specifies the axis
      to be scanned (the sequence axis) for the i-th scan_input. If omitted, 0 will
      be used as the scan axis for every scan_input. Negative value for an axis means
      counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
    required: false
  - name: scan_output_axes
    type: ints
    doc: An optional list of K flags. The i-th element of the list specifies the axis
      for the i-th scan_output. The scan outputs are accumulated along the specified
      axis. If omitted, 0 will be used as the scan axis for every scan_output. Negative
      value for an axis means counting dimensions from the back. Accepted range is
      [-r, r-1].
    required: false
---
