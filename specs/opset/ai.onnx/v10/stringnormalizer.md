---
op: StringNormalizer
domain: ai.onnx
since_version: 10
inputs:
- name: X
  type: tensor(string)
  doc: UTF-8 strings to normalize
outputs:
- name: Y
  type: tensor(string)
  doc: UTF-8 Normalized strings
attributes:
- name: case_change_action
  type: string
  doc: string enum that cases output to be lowercased/uppercases/unchanged. Valid
    values are "LOWER", "UPPER", "NONE". Default is "NONE"
  required: false
  default: NONE
- name: is_case_sensitive
  type: int
  doc: Boolean. Whether the identification of stop words in X is case-sensitive. Default
    is false
  required: false
  default: 0
- name: locale
  type: string
  doc: Environment dependent string that denotes the locale according to which output
    strings needs to be upper/lowercased.Default en_US or platform specific equivalent
    as decided by the implementation.
  required: false
- name: stopwords
  type: strings
  doc: List of stop words. If not set, no word would be removed from X.
  required: false
---

StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
