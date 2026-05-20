---
op: RegexFullMatch
domain: ai.onnx
since_version: 20
type_constraints:
- name: T1
  doc: Inputs must be UTF-8 strings
  allowed:
  - tensor(string)
- name: T2
  doc: Outputs are bools and are True where there is a full regex match and False
    otherwise.
  allowed:
  - tensor(bool)
inputs:
- name: X
  type: T1
  doc: Tensor with strings to match on.
  differentiable: false
outputs:
- name: Y
  type: T2
  doc: Tensor of bools indicating if each input string fully matches the regex pattern
    specified.
  differentiable: false
attributes:
- name: pattern
  type: string
  doc: Regex pattern to match on. This must be valid RE2 syntax.
  required: false
---

RegexFullMatch performs a full regex match on each element of the input tensor. If an element fully matches the regex pattern specified as an attribute, the corresponding element in the output is True and it is False otherwise. [RE2](https://github.com/google/re2/wiki/Syntax) regex syntax is used.
