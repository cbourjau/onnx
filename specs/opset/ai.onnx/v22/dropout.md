---
inherit: v13/dropout
since_version: 22
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
  - name: T1
    doc: Constrain input 'ratio' types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
    - tensor(float8e4m3fn)
    - tensor(float8e4m3fnuz)
    - tensor(float8e5m2)
    - tensor(float8e5m2fnuz)
inputs:
  update:
  - name: ratio
    type: T1
    doc: The ratio of random dropout, with value in [0, 1). If set to 0, the output
      would be a simple copy of the input. If it's non-zero, output will be a random
      dropout of the scaled input, which is typically the case during training. It
      is an optional value, if not specified it will default to 0.5.
    kind: Option
    differentiable: false
---
