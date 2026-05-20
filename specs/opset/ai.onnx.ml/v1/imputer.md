---
op: Imputer
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T
  doc: The input type must be a tensor of a numeric type, either [N,C] or [C]. The
    output type will be of the same tensor type and shape.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: X
  type: T
  doc: Data to be processed.
outputs:
- name: Y
  type: T
  doc: Imputed output data
attributes:
- name: imputed_value_floats
  type: floats
  doc: Value(s) to change to
  required: false
- name: imputed_value_int64s
  type: ints
  doc: Value(s) to change to.
  required: false
- name: replaced_value_float
  type: float
  doc: A value that needs replacing.
  required: false
  default: 0.0
- name: replaced_value_int64
  type: int
  doc: A value that needs replacing.
  required: false
  default: 0
---

Replaces inputs that equal one value with another, leaving all other elements alone.<br>
    This operator is typically used to replace missing values in situations where they have a canonical
    representation, such as -1, 0, NaN, or some extreme value.<br>
    One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    which one depends on whether floats or integers are being processed.<br>
    The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
