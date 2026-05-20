---
inherit: v13/negativeloglikelihoodloss
since_version: 22
type_constraints:
  update:
  - name: T
    doc: Constrain input, weight, and output types to floating-point tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
---
