---
op: LabelEncoder
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input type must be a tensor of integers or strings, of any shape.
  allowed:
  - tensor(int64)
  - tensor(string)
- name: T2
  doc: The output type will be a tensor of strings or integers, and will have the
    same shape as the input.
  allowed:
  - tensor(int64)
  - tensor(string)
inputs:
- name: X
  type: T1
  doc: Input data.
outputs:
- name: Y
  type: T2
  doc: Output data. If strings are input, the output values are integers, and vice
    versa.
attributes:
- name: classes_strings
  type: strings
  doc: A list of labels.
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
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.<br>
    Each operator converts either integers to strings or strings to integers, depending
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    When converting from integers to strings, the string is fetched from the
    'classes_strings' list, by simple indexing.<br>
    When converting from strings to integers, the string is looked up in the list
    and the index at which it is found is used as the converted value.
