---
op: CastMap
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input must be an integer map to either string or float.
  allowed:
  - map(int64, float)
  - map(int64, string)
- name: T2
  doc: The output is a 1-D tensor of string, float, or integer.
  allowed:
  - tensor(float)
  - tensor(int64)
  - tensor(string)
inputs:
- name: X
  type: T1
  doc: The input map that is to be cast to a tensor
outputs:
- name: Y
  type: T2
  doc: A tensor representing the same data as the input map, ordered by their keys
attributes:
- name: cast_to
  type: string
  doc: A string indicating the desired element type of the output tensor, one of 'TO_FLOAT',
    'TO_STRING', 'TO_INT64'.
  required: false
  default: TO_FLOAT
- name: map_form
  type: string
  doc: Indicates whether to only output as many values as are in the input (dense),
    or position the input based on using the key of the map as the index of the output
    (sparse).<br>One of 'DENSE', 'SPARSE'.
  required: false
  default: DENSE
- name: max_map
  type: int
  doc: If the value of map_form is 'SPARSE,' this attribute indicates the total length
    of the output tensor.
  required: false
  default: 1
---

Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
    in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.
