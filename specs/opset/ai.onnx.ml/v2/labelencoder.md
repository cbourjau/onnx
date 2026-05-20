---
inherit: v1/labelencoder
since_version: 2
type_constraints:
  update:
  - name: T1
    doc: The input type is a tensor of any shape.
    allowed:
    - tensor(float)
    - tensor(int64)
    - tensor(string)
  - name: T2
    doc: Output type is determined by the specified 'values_*' attribute.
    allowed:
    - tensor(float)
    - tensor(int64)
    - tensor(string)
inputs:
  update:
  - name: X
    type: T1
    doc: Input data. It can be either tensor or scalar.
outputs:
  update:
  - name: Y
    type: T2
    doc: Output data.
attributes:
  add:
  - name: default_float
    type: float
    doc: A float.
    required: false
    default: -0.0
  - name: keys_floats
    type: floats
    doc: A list of floats.
    required: false
  - name: keys_int64s
    type: ints
    doc: A list of ints.
    required: false
  - name: keys_strings
    type: strings
    doc: A list of strings. One and only one of 'keys_*'s should be set.
    required: false
  - name: values_floats
    type: floats
    doc: A list of floats.
    required: false
  - name: values_int64s
    type: ints
    doc: A list of ints.
    required: false
  - name: values_strings
    type: strings
    doc: A list of strings. One and only one of 'value_*'s should be set.
    required: false
  remove:
  - classes_strings
  update:
  - name: default_int64
    type: int
    doc: An integer.
    required: false
    default: -1
  - name: default_string
    type: string
    doc: A string.
    required: false
    default: _Unused
---

Maps each element in the input tensor to another value.<br>
    The mapping is determined by the two parallel attributes, 'keys_*' and
    'values_*' attribute. The i-th value in the specified 'keys_*' attribute
    would be mapped to the i-th value in the specified 'values_*' attribute. It
    implies that input's element type and the element type of the specified
    'keys_*' should be identical while the output type is identical to the
    specified 'values_*' attribute. If an input element can not be found in the
    specified 'keys_*' attribute, the 'default_*' that matches the specified
    'values_*' attribute may be used as its output value.<br>
    Let's consider an example which maps a string tensor to an integer tensor.
    Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
    and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
    "Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
    Since this operator is an one-to-one mapping, its input and output shapes
    are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
    For key look-up, bit-wise comparison is used so even a float NaN can be
    mapped to a value in 'values_*' attribute.<br>
