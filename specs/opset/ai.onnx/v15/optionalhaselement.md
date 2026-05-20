---
op: OptionalHasElement
domain: ai.onnx
since_version: 15
type_constraints:
- name: O
  doc: Constrain input type to optional tensor and optional sequence types.
  allowed:
  - optional(seq(tensor(bool)))
  - optional(seq(tensor(complex128)))
  - optional(seq(tensor(complex64)))
  - optional(seq(tensor(double)))
  - optional(seq(tensor(float)))
  - optional(seq(tensor(float16)))
  - optional(seq(tensor(int16)))
  - optional(seq(tensor(int32)))
  - optional(seq(tensor(int64)))
  - optional(seq(tensor(int8)))
  - optional(seq(tensor(string)))
  - optional(seq(tensor(uint16)))
  - optional(seq(tensor(uint32)))
  - optional(seq(tensor(uint64)))
  - optional(seq(tensor(uint8)))
  - optional(tensor(bool))
  - optional(tensor(complex128))
  - optional(tensor(complex64))
  - optional(tensor(double))
  - optional(tensor(float))
  - optional(tensor(float16))
  - optional(tensor(int16))
  - optional(tensor(int32))
  - optional(tensor(int64))
  - optional(tensor(int8))
  - optional(tensor(string))
  - optional(tensor(uint16))
  - optional(tensor(uint32))
  - optional(tensor(uint64))
  - optional(tensor(uint8))
- name: B
  doc: Constrain output to a boolean tensor.
  allowed:
  - tensor(bool)
inputs:
- name: input
  type: O
  doc: The optional input.
outputs:
- name: output
  type: B
  doc: A scalar boolean tensor. If true, it indicates that optional-type input contains
    an element. Otherwise, it is empty.
---

Returns true if the optional-type input contains an element. If it is an empty optional-type, this op returns false.
