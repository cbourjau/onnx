---
op: Optional
domain: ai.onnx
since_version: 15
type_constraints:
- name: V
  doc: Constrain input type to all tensor and sequence types.
  allowed:
  - seq(tensor(bool))
  - seq(tensor(complex128))
  - seq(tensor(complex64))
  - seq(tensor(double))
  - seq(tensor(float))
  - seq(tensor(float16))
  - seq(tensor(int16))
  - seq(tensor(int32))
  - seq(tensor(int64))
  - seq(tensor(int8))
  - seq(tensor(string))
  - seq(tensor(uint16))
  - seq(tensor(uint32))
  - seq(tensor(uint64))
  - seq(tensor(uint8))
  - tensor(bool)
  - tensor(complex128)
  - tensor(complex64)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(string)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
- name: O
  doc: Constrain output type to all optional tensor or optional sequence types.
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
inputs:
- name: input
  type: V
  doc: The input element.
  kind: Option
outputs:
- name: output
  type: O
  doc: The optional output enclosing the input element.
attributes:
- name: type
  type: type_proto
  doc: Type of the element in the optional output
  required: false
---

Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
