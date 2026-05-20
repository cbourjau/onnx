---
inherit: v10/roialign
since_version: 16
attributes:
  add:
  - name: coordinate_transformation_mode
    type: string
    doc: Allowed values are 'half_pixel' and 'output_half_pixel'. Use the value 'half_pixel'
      to pixel shift the input coordinates by -0.5 (the recommended behavior). Use
      the value 'output_half_pixel' to omit the pixel shift for the input (use this
      for a backward-compatible behavior).
    required: false
    default: half_pixel
---
