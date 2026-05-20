---
op: CategoryMapper
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input must be a tensor of strings or integers, either [N,C] or [C].
  allowed:
  - tensor(int64)
  - tensor(string)
- name: T2
  doc: The output is a tensor of strings or integers. Its shape will be the same as
    the input shape.
  allowed:
  - tensor(int64)
  - tensor(string)
inputs:
- name: X
  type: T1
  doc: Input data
outputs:
- name: Y
  type: T2
  doc: Output data. If strings are input, the output values are integers, and vice
    versa.
attributes:
- name: cats_int64s
  type: ints
  doc: The integers of the map. This sequence must be the same length as the 'cats_strings'
    sequence.
  required: false
- name: cats_strings
  type: strings
  doc: The strings of the map. This sequence must be the same length as the 'cats_int64s'
    sequence
  required: false
- name: default_int64
  type: int
  doc: An integer to use when an input string value is not found in the map.<br>One
    and only one of the 'default_*' attributes must be defined.
  required: false
  default: -1
- name: default_string
  type: string
  doc: A string to use when an input integer value is not found in the map.<br>One
    and only one of the 'default_*' attributes must be defined.
  required: false
  default: _Unused
---

Converts strings to integers and vice versa.<br>
    Two sequences of equal length are used to map between integers and strings,
    with strings and integers at the same index detailing the mapping.<br>
    Each operator converts either integers to strings or strings to integers, depending
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.
