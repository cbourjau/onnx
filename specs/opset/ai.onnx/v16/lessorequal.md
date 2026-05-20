---
inherit: v12/lessorequal
since_version: 16
function_body: |-
  <
    domain: "",
    opset_import: ["" : 16]
  >
  LessOrEqual (A, B) => (C)
  {
     O1 = Less (A, B)
     O2 = Equal (A, B)
     C = Or (O1, O2)
  }
type_constraints:
  update:
  - name: T
    doc: Constrain input types to all numeric tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(int16)
    - tensor(int32)
    - tensor(int64)
    - tensor(int8)
    - tensor(uint16)
    - tensor(uint32)
    - tensor(uint64)
    - tensor(uint8)
---
